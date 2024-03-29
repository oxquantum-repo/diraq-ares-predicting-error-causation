import numpy as np


def beta_dist(mean, std):
	alpha = (mean ** 2) * (((1 - mean) / (std ** 2)) - 1 / mean)
	beta = alpha * ((1 / mean) - 1)
	return np.random.beta(alpha, beta)

def ravel_index(n, m, shape_m):
    return n * shape_m + m

def full_covariance_matrix_to_spherical(cov):
    return np.array([cov[0, 0], cov[1, 1]])