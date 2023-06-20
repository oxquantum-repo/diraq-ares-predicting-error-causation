import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from time import time

from errorcausation import GaussianModel
from scipy.stats import norm


np.random.seed(42)

decision_boundary = 0.5

model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.5, 0.5])

model.transmat_ = np.array([[0.99, 0.01],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1., 0.]])
model.covars_ = np.array([0.4, 0.4]) ** 2

t0 = time()
N = 10
X, Z = model.simulate_data(N, repeats = 1000)
t1 = time()
print(f"Time to simulate data: {t1 - t0:.3f}s")

I = X[..., 0].flatten()
Q = X[..., 1].flatten()

t0 = time()
model_to_fit = GaussianModel(n_components=2, covariance_type="spherical")
model_to_fit.fit(X)

predictions = model_to_fit.predict(X)

I_ground = I[predictions.flatten() == 0]
I_excited = I[predictions.flatten() == 1]

Q_ground = Q[predictions.flatten() == 0]
Q_excited = Q[predictions.flatten() == 1]

t1 = time()
print(f"Time to initialize and fit model: {t1 - t0:.3f}s")

print(f"startprob:  \n{model_to_fit.startprob_}")
print(f"transmat: \n{model_to_fit.transmat_}")

print(f"means: \n{model_to_fit.means_}")
print(f"std: \n{model_to_fit.covar_to_std()}")

p_10 = np.mean(I_ground > decision_boundary)
p_01 = np.mean(I_excited < decision_boundary)

if p_10 > 0.5 and p_01 > 0.5:
    p_10 = 1 - p_10
    p_01 = 1 - p_01

print(f"Frequentest p(1|0), p(0|1) = {p_01:.3f}, {p_10:.3f}")

I_ground_mean = model_to_fit.means_[:, 0]
I_ground_std = model_to_fit.covar_to_std()[0]

p_10_analytical = min(norm(model_to_fit.means_[:, 0], model_to_fit.covar_to_std()[0]).cdf(decision_boundary))
p_01_analytical = 1 - min(norm(model_to_fit.means_[:, 1], model_to_fit.covar_to_std()[1]).cdf(decision_boundary))

print(f"Bayesian p(1|0), p(0,1) = {p_10_analytical:.3f}, {p_01_analytical:.3f}")

IQ_range = max([I.max() - I.min(), Q.max() - Q.min()])
I_bins = np.linspace(I.mean() -IQ_range / 2, I.mean() + IQ_range / 2, 40)
Q_bins = np.linspace(Q.mean() -IQ_range / 2, Q.mean() + IQ_range / 2, 40)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.set_size_inches(5, 2.5)
ax[0].scatter(I, Q, s = 1)
ax[0].hist2d(I, Q, cmap='hot', bins = [I_bins, Q_bins])


for i in range(model_to_fit.n_components):
    std = model_to_fit.covar_to_std()[i]
    ax[0].add_artist(plt.Circle(model_to_fit.means_[i], 2 * std, color='b', fill=False))
ax[0].set_xlabel('I')
ax[0].set_ylabel('Q')
ax[0].set_xlim(I_bins[0], I_bins[-1])
ax[0].set_ylim(Q_bins[0], Q_bins[-1])
ax[0].set_aspect('equal')
ax[0].axvline(decision_boundary, color='w', linestyle='--')

ax[1].scatter(I_ground, Q_ground, s = 1, label='ground')
ax[1].scatter(I_excited, Q_excited, s = 1, label='excited')
ax[1].set_xlabel('I')
ax[1].set_ylabel('Q')
ax[1].set_aspect('equal')
ax[1].axvline(decision_boundary, color='k', linestyle='--')


