import matplotlib.pyplot as plt
import numpy as np
from time import time

from errorcausation import GaussianModel
from errorcausation.opx.gaussian_hmm_algorithms_raw_python import forward


model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.8, 0.2])

model.transmat_ = np.array([[0.97, 0.03],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1., 1.]]) / np.sqrt(2)
model.covars_ = np.array([0.3, 0.3]) ** 2

X, Z = model.simulate_data(10, repeats = 1000, plot=False)

model_to_fit = GaussianModel(n_components=2, covariance_type="spherical")
model_to_fit.fit(X)
readout_data = model_to_fit.readout_fidelity(X, plot = True)
print(f'Fidelity: {readout_data.f_ground :.3f}, {readout_data.f_excited :.3f}')

t0 = time()
N = 100
X, Z = model.simulate_data(N, repeats = 1, plot=True)

I = X[..., 0].flatten()
Q = X[..., 1].flatten()

alpha = forward(X,
    np.array([0.5, 0.5]),
    model_to_fit.transmat_,
    model_to_fit.means_,
    model_to_fit.covars_
)

plt.figure()
plt.plot(alpha[:, 0], label="I")
plt.plot(alpha[:, 1], label="I")
plt.yscale("log")
plt.show()