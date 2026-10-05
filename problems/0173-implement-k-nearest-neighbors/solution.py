import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points_arr = np.array(points)
    dists = np.linalg.norm(points_arr - query_point, axis=1) # axis = -1?
    sorted_k_idx = np.argsort(dists)[:k]
    res = list(map(tuple, points_arr[sorted_k_idx]))
    return res
    
    
