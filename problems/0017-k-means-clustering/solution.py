import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	points = np.array(points, dtype=float) # n, d
	n, d = points.shape
	centroids = np.array(initial_centroids) # k, d
	for _ in range(max_iterations):
		dist_matrix = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=-1) # (n, k)
		# min_k = dist_matrix.min(axis=0) #(_, n), does the dim collapase
		centroids = compute_centroids(dist_matrix, centroids, k, points, n)

	return [tuple(c) for c in centroids.tolist()]

	#return centroids

def compute_centroids(dist_matrix, centroids, k, points, n):
	new_centroids = np.zeros_like(centroids, dtype=np.float32)
	centroid_idxs = np.argmin(dist_matrix, axis=-1) #(n,) -> idx of the closest centroid per n in the dist matrix
	for i in range(k):
		# create a bool mask
		selected_idxs = (centroid_idxs == i) # (n,) closets k is i
		# print("selected_idxs", selected_idxs)
		# members_k_idx = centroid_idxs[selected_idxs] # how do do this?? 
		# print("n", n, "points.n", points[members_k_idx].shape[0], "points.d", points[members_k_idx].shape[1])
		# print("points", points, "selected", points[members_k_idx])
		new_center = points[selected_idxs].mean(axis=0) # (n, d) -> (_, d)
		# print(new_center)
		new_centroids[i] = new_center
	return new_centroids

