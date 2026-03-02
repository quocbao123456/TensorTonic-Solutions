import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cos = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

    return  1 - cos if label == 1 else max(0, cos - margin)