from .data_utils_mutilsource import gen_batch
from .math_utils_mutilsource import evaluation
from os.path import join as pjoin
import numpy as np
import time
import torch


def multi_pred(device, model, seq, batch_size, n_his, n_pred, dynamic_batch=True):
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq[0])), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his, :, :, :])
        step_list = []
        test_seq_th = torch.tensor(test_seq, dtype=torch.float32).to(device)
        pred = model(test_seq_th)
        pred = pred.data.cpu().numpy()
        pred_list.append(pred)
    #  pred_array -> [batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def model_inference(device,model, inputs, batch_size, n_his, n_pred, min_va_val, min_val, n, scaler):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    if n_his + n_pred > x_val[0].shape[0]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')
    y_val, len_val = multi_pred(device,model, x_val, batch_size, n_his, n_pred)
    mape_val, mae_val, rmse_val= evaluation(x_val[0:len_val, n_his:n_pred + n_his, : n, :, :], y_val[:, :, : n, :, :], x_stats, scaler)
    # update the metric on test set, if model's performance got improved on the validation.
    y_test, len_test = multi_pred(device,model, x_test, batch_size, n_his, n_pred)
    mape_test, mae_test, rmse_test = evaluation(x_test[0:len_test, n_his:n_pred + n_his, : n, :, :], y_test[:, :, : n, :, :], x_stats, scaler)
    return mape_val, mae_val, rmse_val, mape_test, mae_test, rmse_test
