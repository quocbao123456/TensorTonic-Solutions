import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape

    w = np.zeros(n)
    b = 0.0

    for _ in range(steps):
        #Forward
        z = np.dot(X, w) + b
        y_hat = _sigmoid(z)

        # Compute loss
        loss = -np.mean(
            y*np.log(y_hat) +
            (1 - y)*np.log(1 - y_hat)
         )
        
        #Compute gradient
        error = y_hat - y
        dw = np.dot(X.T, error)/m
        db = np.sum(error)/m


        #Update gradient
        w -= lr*dw
        b -= lr*db
    return w, b
    
        # dw = (1/m) * (X.T @ error) 
        # db = (1/m) * np.sum(error