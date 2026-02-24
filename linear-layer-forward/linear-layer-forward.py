def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X = np.array(X, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)

    return (np.dot(X,W) + b).tolist()
    # Write code here