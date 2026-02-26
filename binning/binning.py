import numpy as np
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """

    values = np.array(values)
    maxX = np.max(values)
    minX = np.min(values)

    w = (maxX - minX)/num_bins
    if w == 0:
        return np.zeros(len(values), dtype=int).tolist()


    return np.array([min(int((x - minX) / w), num_bins - 1) for x in values]).tolist()
