import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.array(A)
    
    # return np.sum(np.diag(A))
    return sum(A[i, i] for i in range(A.shape[0]))
