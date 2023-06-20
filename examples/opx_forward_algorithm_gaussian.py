import numpy as np
from time import time

from errorcausation import GaussianModel


np.random.seed(42)

decision_boundary = 0.5

model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.5, 0.5])

model.transmat_ = np.array([[0.9, 0.1],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1., 0.]])
model.covars_ = np.array([0.1, 0.1]) ** 2

t0 = time()
N = 40
X, Z = model.simulate_data(N, repeats = 1)

I = X[..., 0].flatten()
Q = X[..., 1].flatten()