import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import Parameter
import math
import sys
import numpy as np
from torchsummary import summary
import warnings

warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

LOG2PI = math.log(2 * math.pi)

class GMRE(nn.Module):
    def __init__(self, device, num_comp, channels, num_nodes, num_source, n_his, dilation):
        super(GMRE, self).__init__()
        self.device = device
        self.time = n_his - dilation + 1
        self.in_features = num_nodes * num_source * self.time
        self.num_cluster = num_comp
        self.alpha = nn.Sequential(
            nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1)
        self.mu = nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1)
    
    def getlogPdf(self, target, mu, sigma):
        _,_,N = target.shape
        _,_,k = sigma.shape
        target = target.unsqueeze(-1).repeat((1,1,1,k))
        mu = mu.unsqueeze(2).repeat((1,1,N,1))
        sigma = sigma.unsqueeze(2).repeat((1,1,N,1))
        log_component_prob = (-torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - mu) / sigma, 2))   
        return log_component_prob
    
    def calculate_loss(self, alphas, log_component_prob, weightedlogPdf):
        first_item = torch.mean(weightedlogPdf.exp() * log_component_prob)
        q_z = weightedlogPdf.mean((0,2))
        KL = F.kl_div(q_z, alphas, reduction='mean')
        loss =  KL - first_item
        return loss        
        
    def cluster_norm(self, mu, sigma, target, labels, num_cluster):
        new_mu = labels
        new_sigma = labels        
        for k in range(num_cluster):
            mask_k = (labels==k)
            mu_k = mu[:,:,k].unsqueeze(-1).expand_as(mask_k)
            sigma_k = sigma[:,:,k].unsqueeze(-1).expand_as(mask_k)
            new_mu[mask_k] = mu_k[mask_k]
            new_sigma[mask_k] = sigma_k[mask_k]
        return (target - new_mu) / (new_sigma + 0.00001)
    
    def forward(self, x):
        b, c, s, n, t = x.shape
        hd = x.reshape(b, c, -1).permute(0,2,1)  
        alphas = self.alpha(hd).permute(0,2,1)    
        sigma = torch.exp(self.sigma(hd)) 
        sigma = sigma.permute(0,2,1)
        mu = self.mu(hd)   
        mu = mu.permute(0,2,1)
        hd = hd.permute(0,2,1) 
        
        # get PDF and labels
        log_component_prob = self.getlogPdf(hd, mu, sigma)  
        weightedlogPdf = log_component_prob + torch.log(alphas.unsqueeze(2))           
        labels = torch.argmax(weightedlogPdf,dim=-1).float()  
        # cluster normalization
        norm = self.cluster_norm(mu, sigma, hd, labels, self.num_cluster)  
        out = norm.reshape(b,c,s,n,t)
        # get loss
        loss = self.calculate_loss(alphas, log_component_prob, weightedlogPdf)
        
        return out, loss    

class ResidualBlock(nn.Module):
    def __init__(self, device, num_comp, num_nodes, num_source, n_pred, n_his, channels, dilation, kernel_size, hra_cell):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.num_source = num_source
        self.dilation = dilation
        self.hra_cell = hra_cell
        self.gmre_cell = nn.ModuleList()
        # N cells GMRE
        for i in range(self.hra_cell):
            self.gmre_cell.append(GMRE(self.device, num_comp, channels, num_nodes, num_source, n_his, self.dilation))
        
        self.num = self.hra_cell + 1         
        # Temporal Encoder
        self.filter_convs = nn.Conv3d(in_channels = self.num * channels, 
                                      out_channels = num_source * channels, 
                                      kernel_size = (num_source, 1, kernel_size),
                                      dilation=(1,1,self.dilation))
        self.gate_convs = nn.Conv3d(in_channels = self.num * channels, 
                                    out_channels = num_source * channels, 
                                    kernel_size = (num_source, 1, kernel_size),
                                    dilation=(1,1,self.dilation))
        self.residual_convs = nn.Conv3d(in_channels = channels, out_channels = channels, kernel_size = (1,1,1))
        # Skip Connection
        self.skip_convs = nn.Conv3d(in_channels = channels, out_channels = channels, kernel_size = (1,1,1))

    def forward(self, x):
        x_list = []
        loss_list = []
        # GMRE-Cell
        for i in range(self.hra_cell):
            x_gmre, feature_loss = self.gmre_cell[i](x)
            x_list.append(x_gmre)
            loss_list.append(feature_loss)
        x_list.append(x)
        x = torch.cat(x_list, dim=1) 
        loss = torch.Tensor(loss_list).mean()
        # Temporal Encoder
        filter = self.filter_convs(x)
        b, _, _, n, t = filter.shape
        filter = torch.tanh(filter).reshape(b, -1, self.num_source, n, t)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate).reshape(b, -1, self.num_source, n, t)
        x = filter * gate
        # parametrized skip connection
        save_x = x
        sk = x
        sk = self.skip_convs(sk)
        x = self.residual_convs(x)
        return x, sk, gate, loss

class HRA(nn.Module):
    def __init__(self, device, num_nodes, num_source, n_pred, out_dim,channels, hra_bool):
        super(HRA, self).__init__()
        self.hra_bool = hra_bool
        
        if self.hra_bool:
            # construct memory
            self.memo_num = 8
            self.memo_dim = channels
            self.flat_hidden = num_source * num_nodes * channels
            self.W_q = nn.Parameter(torch.randn(self.flat_hidden, self.memo_dim), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.W_q)
            self.memory = nn.Parameter(torch.randn(self.memo_num, self.memo_dim), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.memory)
            self.W_fc = nn.Parameter(torch.randn(self.memo_dim, self.flat_hidden), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.W_fc)
            # Predictor
            self.outlayer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels = channels + self.memo_dim, out_channels = channels,kernel_size = (1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels = channels, out_channels = n_pred * out_dim, kernel_size = (1,1,1))
            )
        else:
            # Predictor
            self.outlayer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels = channels,out_channels = channels,kernel_size = (1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels = channels, out_channels = n_pred * out_dim, kernel_size = (1,1,1))
            )
    def forward(self, hd):
        b, c, s, n, t = hd.shape
        if self.hra_bool:
            ht_flat = hd.reshape(b,-1)
            query = torch.mm(ht_flat, self.W_q)
            att_score = torch.softmax(torch.mm(query, self.memory.t()), dim=1)
            att_memory = torch.mm(torch.mm(att_score, self.memory), self.W_fc)      
            att_memory = att_memory.reshape(b,c,s,n,t)
            h_con = torch.cat([hd, att_memory], dim=1)
            out = self.outlayer(h_con)
        else:
            out = self.outlayer(hd)
        return out
    
class GMRL(nn.Module):
    def __init__(self, device, num_comp, num_nodes, num_source, n_his, n_pred, in_dim=1, out_dim=1, channels=16, kernel_size=2,layers=2,hra_cell=3,hra_bool=True):
        super(GMRL, self).__init__()
        self.layers = layers
        self.in_dim = in_dim
        self.hra_bool = hra_bool
        # Linear Projection
        self.proj = nn.Conv3d(in_channels = in_dim, out_channels = channels, kernel_size = (1,1,1))
        # Embedding Layer
        self.temporal_embedding = nn.Parameter(torch.randn(channels, n_his), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.temporal_embedding)
        self.location_embedding = nn.Parameter(torch.randn(channels, num_nodes), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.location_embedding)
        self.source_embedding = nn.Parameter(torch.randn(channels, num_source), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.source_embedding)
        # Residual Blocks
        self.residualblocks = nn.ModuleList()
        dilation = 1
        for i in range(self.layers):
            self.residualblocks.append(ResidualBlock(device, num_comp, num_nodes, num_source, n_pred, n_his, 2*channels, dilation, kernel_size, hra_cell))
            dilation *= 2
        # Predictor with Hidden Representation Augmenter
        self.hra = HRA(device, num_nodes, num_source, n_pred, out_dim, 2*channels, self.hra_bool)

    def forward(self, input):
        input = input.permute(0, 4, 3, 2, 1) 
        # Init representation
        x = self.proj(input)
        b,c,s,n,t = x.shape
        # Embedding Layer
        tts_embeddings = self.temporal_embedding.reshape(1,c,1,1,t) + self.location_embedding.reshape(1,c,1,n,1) + self.source_embedding.reshape(1,c,s,1,1)
        tts_embeddings = tts_embeddings.repeat(b,1,1,1,1)
        x = torch.cat((x, tts_embeddings), dim=1)
        # Hidden layers
        skip = 0        
        for i in range(self.layers):           
            residual = x
            x, sk, gate, feature_loss = self.residualblocks[i](x)
            x = x + residual[:, :, :, :, -x.size(4):]
            try:
                skip = sk + skip[:, :, :, :, -sk.size(4):]
            except:
                skip = sk        
        # generate HRA-augmented pediction         
        out = self.hra(skip)
        out = out.permute(0, 1, 3, 2, 4)
        return out, feature_loss

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)
