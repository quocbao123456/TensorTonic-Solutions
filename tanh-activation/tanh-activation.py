import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x)
    e = np.exp(x)
    e1 = np.exp(-x)
    
    return (e - e1)/(e + e1)