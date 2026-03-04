import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """

    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    if x.ndim == 2:
        mean = np.mean(x, axis=0)
        variance = np.sum((x - mean)**2, axis=0) / x.shape[0]
    
        norm = (x - mean)/np.sqrt(variance + eps)
    
        return gamma*norm + beta
    else:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        variance = np.var(x, axis=(0, 2, 3), keepdims=True)
    
        norm = (x - mean)/np.sqrt(variance + eps)
        
        gamma = gamma.reshape(1,-1,1,1)
        beta = beta.reshape(1,-1,1,1)
    
        return gamma*norm + beta