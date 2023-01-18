import numpy as np

def z_score(x, mean, std):
    return (x - mean) / std


def z_inverse(x, mean, std):
    return x * std + mean


def MAPE(y_true, y_pred,null_val=0):
    y_true[y_true < 20] = 0
    y_pred[y_pred < 20] = 0    
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
    return np.mean(mape, axis=(0, 2, 4))


def RMSE(v, v_):
    return np.mean((v_ - v) ** 2, axis=(0, 2, 4)) ** 0.5   

def MAE(v, v_):
    return np.mean(np.abs(v_ - v), axis=(0, 2, 4))


def evaluation(y, y_, x_stats, scaler):
    dim = len(y_.shape)
    if scaler == "global_scaler":
        y = z_inverse(y, x_stats['mean'], x_stats['std'])
        y_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    else:
        y = z_inverse(y, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
        y_ = z_inverse(y_, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
    mape = MAPE(y, y_)
    rmse = RMSE(y, y_)
    mae = MAE(y, y_)
    return mape, mae, rmse
