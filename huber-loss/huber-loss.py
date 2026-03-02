import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error = y_true - y_pred
    abs_error = np.abs(error)

    quadaric = 0.5*error**2
    linear = delta*(abs_error - delta/2)
    
    loss = np.where(abs_error <= delta, quadaric, linear)

    return np.mean(loss)
    