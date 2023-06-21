import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from time import time

from errorcausation import GaussianModel, readout_fidelity, readout_corrections
from scipy.stats import norm


model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.9, 0.1])

model.transmat_ = np.array([[0.99, 0.01],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1., 1.]]) / np.sqrt(2)
model.covars_ = np.array([0.2, 0.2]) ** 2

t0 = time()
N = 20
X, Z = model.simulate_data(N, repeats = 1000)
t1 = time()
print(f"Time to simulate data: {t1 - t0:.3f}s")

t0 = time()
model_to_fit = GaussianModel(n_components=2, covariance_type="spherical")
model_to_fit.fit(X)
readout_fidelity_data = model_to_fit.readout_fidelity(X, plot=True)

t1 = time()
print(f"Time to initialize and fit model: {t1 - t0:.3f}s")

print(f"startprob:  \n{model_to_fit.startprob_}")
print(f"transmat: \n{model_to_fit.transmat_}")

print(f"means: \n{model_to_fit.means_}")
print(f"std: \n{model_to_fit.covar_to_std()}")






