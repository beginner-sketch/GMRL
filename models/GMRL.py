import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

LOG2PI = math.log(2 * math.pi)

class GMRE(nn.Module):
    ''' Gaussian Mixture Representation Extractor (GMRE) '''
    def __init__(self, device, num_comp, channels, num_nodes, num_source, n_his, dilation):
        super(GMRE, self).__init__()
        self.device = device
        self.time = n_his - dilation + 1
        self.in_features = num_nodes * num_source * self.time
        # The number of clusters K
        self.num_cluster = num_comp
        self.alpha = nn.Sequential(
            nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1)
        self.mu = nn.Conv1d(in_channels = self.in_features, out_channels = self.num_cluster, kernel_size = 1)
    
    def getlogPdf(self, target, mu, sigma):
        ''' Get Probability Density Functions (PDFs) : N(h|\mu_k(c),\sigma^2_k(c)) 
            target -> (batch, channel, N=T*L*S)
            mu -> (batch, channel, K)
            sigma -> (batch, channel, K)
            log_component_prob -> (batch, channel, N=T*L*S, K)
        '''
        _,_,N = target.shape
        _,_,k = sigma.shape
        target = target.unsqueeze(-1).repeat((1,1,1,k))
        mu = mu.unsqueeze(2).repeat((1,1,N,1))
        sigma = sigma.unsqueeze(2).repeat((1,1,N,1))
        log_component_prob = (-torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - mu) / sigma, 2))   
        return log_component_prob
    
    def calculate_loss(self, alphas, log_component_prob, weightedlogPdf):
        ''' Probability Regularization '''
        first_item = torch.mean(weightedlogPdf.exp() * log_component_prob)
        # Calculate the Kullback–Leibler divergence: KL( Q(z|c) || P(z|c))
        q_z = weightedlogPdf.mean(2)
        KL = F.kl_div(q_z, alphas, reduction='mean')
        loss =  torch.mean(KL) - first_item
        return loss        
        
    def cluster_norm(self, mu, sigma, target, labels, num_cluster):
        ''' Cluster Normalization 
        target, labels -> (batch, channel, N=T*L*S)
        mu, sigma -> (batch, channel, K)
        norm -> (batch, channel, N=T*L*S)
        '''
        _,c,N = target.shape
        # Initialize a new mu and sigma with the same shape as the target
        new_mu = labels
        new_sigma = labels
        for k in range(num_cluster):
            # mask_k: obtain the position equal to the current cluster number (k) in the labels
            mask_k = (labels==k)
            # mu_k and sigma_k: the subsets of corresponding clusters in the mu and sigma respectively
            mu_k = mu[:,:,k].unsqueeze(-1).expand_as(mask_k)
            sigma_k = sigma[:,:,k].unsqueeze(-1).expand_as(mask_k)
            # Cover new_mu and new_sigma with mu_k and sigma_k respectively
            new_mu[mask_k] = mu_k[mask_k]
            new_sigma[mask_k] = sigma_k[mask_k]
        # Cluster-wise normalization
        norm = (target - new_mu) / (new_sigma + 0.00001)
        return norm
    
    def forward(self, x):
        b, c, s, n, t = x.shape
        hd = x.reshape(b, c, -1).permute(0,2,1)  
        # Get alphas (P(z = k|c)) with shape (batch, channel, K)
        alphas = self.alpha(hd).permute(0,2,1)    
        # Get mu and sigma with shape (batch, channel, K)
        sigma = torch.exp(self.sigma(hd)) 
        sigma = sigma.permute(0,2,1)
        mu = self.mu(hd)   
        mu = mu.permute(0,2,1)
        hd = hd.permute(0,2,1) 
        # Get PDF with shape (batch, channel, N=T*L*S, k) (convert to log form for easy calculation)
        log_component_prob = self.getlogPdf(hd, mu, sigma)  # log_component_prob: log(P(h|z = k,c))
        log_prob = log_component_prob + torch.log(alphas.unsqueeze(2))  # log_prob: log(P(h|z = k,c)P(z = k|c))
        log_sum = torch.sum(log_prob.exp(), dim=-1, keepdim=True)        
        log_sum = torch.log(log_sum)    # log_sum: log(\sum_k P(h|z = k,c)P(z = k|c))
        weightedlogPdf = log_prob - log_sum     # weightedlogPdf: log(P(z = k|h,c))
        labels = torch.argmax(weightedlogPdf,dim=-1).float()    # labels: argmax_k P(z = k|h,c)
        # Cluster-wise normalization
        norm = self.cluster_norm(mu, sigma, hd, labels, self.num_cluster)  
        out = norm.reshape(b,c,s,n,t)
        # Get L^(cluster)
        loss = self.calculate_loss(alphas, log_component_prob, weightedlogPdf)
        return out, loss    

class ResidualBlock(nn.Module):
    def __init__(self, device, num_comp, num_nodes, num_source, n_pred, n_his, channels, dilation, kernel_size):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.num_source = num_source
        self.dilation = dilation
        self.gmre = nn.ModuleList()
        # GMRE heads
        self.gmre = GMRE(self.device, num_comp, channels, num_nodes, num_source, n_his, self.dilation)
        
        self.num = 2         
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
        # GMRE heads
        x_gmre, feature_loss = self.gmre(x)
        x_list.append(x_gmre)
        x_list.append(x)
        x = torch.cat(x_list, dim=1) 
        loss = feature_loss
        # Temporal Encoder (TE)
        filter = self.filter_convs(x)
        b, _, _, n, t = filter.shape
        filter = torch.tanh(filter).reshape(b, -1, self.num_source, n, t)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate).reshape(b, -1, self.num_source, n, t)
        x = filter * gate
        # Parametrized skip connection
        save_x = x
        sk = x
        sk = self.skip_convs(sk)
        x = self.residual_convs(x)
        return x, sk, gate, loss

class HRA(nn.Module):
    ''' Hidden Representation Augmenter (HRA) '''
    def __init__(self, device, num_nodes, num_source, n_pred, out_dim,channels, hra_bool):
        super(HRA, self).__init__()
        self.hra_bool = hra_bool
        
        if self.hra_bool:
            self.memo_num = 8
            self.memo_dim = channels
            self.flat_hidden = num_source * num_nodes * channels
            # Construct memory M with shape (memo_num, memo_dim)
            self.memory = nn.Parameter(torch.randn(self.memo_num, self.memo_dim), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.memory)
            # Construct weight matrix W_q with shape (flat_hidden, memo_dim)
            self.W_q = nn.Parameter(torch.randn(self.flat_hidden, self.memo_dim), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.W_q)
            # Construct weight matrix W_fc with shape (memo_dim, flat_hidden)
            self.W_fc = nn.Parameter(torch.randn(self.memo_dim, self.flat_hidden), requires_grad=True).to(device)
            nn.init.xavier_normal_(self.W_fc)
            # Predictor with HRA-augmented representation
            self.outlayer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels = channels + self.memo_dim, out_channels = channels,kernel_size = (1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels = channels, out_channels = n_pred * out_dim, kernel_size = (1,1,1))
            )
        else:
            # Predictor without HRA-augmented representation
            self.outlayer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels = channels,out_channels = channels,kernel_size = (1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels = channels, out_channels = n_pred * out_dim, kernel_size = (1,1,1))
            )
    def forward(self, h_sc):
        b, c, s, n, t = h_sc.shape
        if self.hra_bool:
            ht_flat = h_sc.reshape(b,-1)
            query = torch.mm(ht_flat, self.W_q)     # query vector projected from flatted representation with shape (b, memo_dim)
            att_score = torch.softmax(torch.mm(query, self.memory.t()), dim=1)  # attention score with shape (b, memo_num)
            V = torch.mm(att_score, self.memory)    # reconstructed prototype representation with shape (b, memo_dim)
            h_me = torch.mm(V, self.W_fc)      
            h_me = h_me.reshape(b,c,s,n,t)
            h_aug = torch.cat([h_sc, h_me], dim=1)
            out = self.outlayer(h_aug)
        else:
            out = self.outlayer(h_sc)
        return out
    
class GMRL(nn.Module):
    ''' Gaussian Mixture Representation Learning (GMRL) '''
    def __init__(self, device, num_comp, num_nodes, num_source, n_his, n_pred, in_dim=1, out_dim=1, channels=16, kernel_size=2,layers=2,hra_bool=True):
        super(GMRL, self).__init__()
        # The number of GMRE-TE layers
        self.layers = layers
        self.in_dim = in_dim
        self.hra_bool = hra_bool
        # Linear Projection
        self.proj1 = nn.Conv3d(in_channels = in_dim, out_channels = channels, kernel_size = (1,1,1))
        # Tensor Time Series Embedding (TTSE)
        self.temporal_embedding = nn.Parameter(torch.randn(channels, n_his), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.temporal_embedding)
        self.location_embedding = nn.Parameter(torch.randn(channels, num_nodes), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.location_embedding)
        self.source_embedding = nn.Parameter(torch.randn(channels, num_source), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.source_embedding)
        # Linear Projection v3
        self.proj3 = nn.Sequential(
            nn.Conv3d(in_channels = 2 * channels, out_channels = 2 * channels, kernel_size = (1,1,1)),
            nn.ReLU())
        # Residual Blocks
        self.residualblocks = nn.ModuleList()
        dilation = 1
        for i in range(self.layers):
            self.residualblocks.append(ResidualBlock(device, num_comp, num_nodes, num_source, n_pred, n_his, 2*channels, dilation, kernel_size))
            dilation *= 2
        # Predictor with HRA
        self.hra = HRA(device, num_nodes, num_source, n_pred, out_dim, 2*channels, self.hra_bool)

    def forward(self, input):
        input = input.permute(0, 4, 3, 2, 1) 
        # Init representation
        x = self.proj1(input)
        b,c,s,n,t = x.shape
        # Calculate TTSE
        ttse = self.temporal_embedding.reshape(1,c,1,1,t) + self.location_embedding.reshape(1,c,1,n,1) + self.source_embedding.reshape(1,c,s,1,1)
        ttse = ttse.repeat(b,1,1,1,1)
        # Linear Projection: 
        x = self.proj3(torch.cat((x, ttse), dim=1))
        # GMRE-TE layers
        skip = 0        
        for i in range(self.layers):           
            residual = x
            x, sk, gate, feature_loss = self.residualblocks[i](x)
            x = x + residual[:, :, :, :, -x.size(4):]
            try:
                skip = sk + skip[:, :, :, :, -sk.size(4):]
            except:
                skip = sk        
        # Generate HRA-augmented pediction 
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
