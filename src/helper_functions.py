import numpy as np


def beta_distribution_from_mean_and_std(mean, std):
	alpha = (mean ** 2) * (((1 - mean) / (std ** 2)) - 1 / mean)
	beta = alpha * ((1 / mean) - 1)
	return np.random.beta(alpha, beta)