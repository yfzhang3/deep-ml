import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	n, d = np.array(vectors).shape
	# print(n, d)
	norm_vect = vectors - np.mean(vectors, axis=1, keepdims=True)
	return ((norm_vect @ norm_vect.T)/(d-1)).tolist()