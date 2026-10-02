import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_T = np.array(X).T
	return np.linalg.inv(X_T @ X) @ X_T @ y