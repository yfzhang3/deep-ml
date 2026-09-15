import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	if np.prod(new_shape) != np.size(a):
		return []
	arr = np.reshape(a, new_shape)
	reshaped_matrix = arr.tolist()
	return reshaped_matrix