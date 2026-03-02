import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    result = 0.0
    for index, p in enumerate(y_pred):
        index_true = y_true[index]

        result -= np.log(p[index_true])

    result /= len(y_true)

    return result