import numpy as np

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    matrix = np.array(matrix, dtype=float)

    user_sum = np.sum(matrix, axis=1, keepdims=True)
    user_count = np.count_nonzero(matrix, axis=1, keepdims=True)

    user_mean = user_sum / user_count

    return np.where(matrix, matrix - user_mean, 0).tolist()