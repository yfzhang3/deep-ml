
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	n = len(y)
	y = np.array(y)
	c = len(np.unique(y))
	p_sum = 0.0
	for i in range(c):
		p_i = (np.sum((y==i).astype(float))/n)**2
		p_sum += p_i
	val = 1 - p_sum
	return round(val,3)