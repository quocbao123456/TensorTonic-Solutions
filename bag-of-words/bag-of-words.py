import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    tokens = np.array(tokens)
    if len(tokens) == 0:
        return np.zeros(len(vocab), dtype=int)
    values, freqs = np.unique(tokens, return_counts=True)

    freq_map = dict(zip(values, freqs))

    return np.array([freq_map.get(val, 0) for val in vocab], dtype=int)
