import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.array(predictions)
    modes = []
    
    for col in range(predictions.shape[1]):
        values, counts = np.unique(predictions[:,col], return_counts=True)

        mode = values[np.argmax(counts)]
        modes.append(mode)

    return modes