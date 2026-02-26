import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)
    centroids = np.array(centroids)

    result = np.zeros(len(points), dtype=int)

    for index, point in enumerate(points):
        distances = np.linalg.norm(centroids - point, axis=1)

        result[index] = np.argmin(distances)
    
    return result.tolist()