import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError()

    # return np.dot(x, y)
    return sum(a*b for a, b in zip(x, y))
