import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
	"""
	Initialize k centroids using the K-Means++ algorithm.
	
	Args:
		X: Data points of shape (n_samples, n_features)
		k: Number of centroids to initialize
		seed: Random seed for reproducibility
	
	Returns:
		Centroids of shape (k, n_features)
	"""
	n, d = X.shape
	np.random.seed(seed)

	centroids = np.zeros((k, d))
	all_min_sq_dist_diff = np.full((n,), np.inf) # init all to inf
	probs = np.full((n,), np.inf)

	for i in range(k):
		if i ==0: 
			first_idx = np.random.randint(0, n)
			centroids[i] = X[first_idx]
			all_min_sq_dist_diff[first_idx] = 0
		else:
			sq_dist_from_centroid = np.linalg.norm(X[:, None, :] - (centroids[None, :, :])[:,:i,:], axis=-1)**2
			# (n, _, d) - (k (through i), d) -> (n, k, d) -> (n, k)
			# min_k_sq_dist = np.argmin(sq_dist, axis=-1) # (n,)
			# chosen_k = majority_vote(min_k_sq_dist) # (int)
			min_sq_dist_diff = np.min(sq_dist_from_centroid, axis=-1) # (n,)
			all_min_sq_dist_diff = np.minimum(min_sq_dist_diff, all_min_sq_dist_diff)
			# print("shape", min_sq_dist_diff.shape)
			# print(all_min_sq_dist_diff.shape)
			probs = all_min_sq_dist_diff / sum(all_min_sq_dist_diff)
			new_centroid = np.random.choice(np.arange(n), p=probs)
			centroids[i] = X[new_centroid]
	
	return centroids
