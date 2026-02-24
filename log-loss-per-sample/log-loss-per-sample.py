import numpy as np

def log_loss(y_true, y_pred, eps=1e-15):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    p = np.clip(y_pred, eps, 1 - eps)

    return (-y_true*np.log(p) - (1-y_true)*np.log1p(-p)).tolist()