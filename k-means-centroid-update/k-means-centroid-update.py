import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.array(points)
    assignments = np.array(assignments)

    result = np.zeros((k, points.shape[1]))

    for i in range(k):
        cluster_points = points[assignments == i]

        if len(cluster_points) > 0:
            result[i] = np.mean(cluster_points, axis=0)
    return result.tolist()
    