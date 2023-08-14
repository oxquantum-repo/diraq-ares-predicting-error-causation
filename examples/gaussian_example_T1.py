import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from time import time

from errorcausation import GaussianModel

np.random.seed(42)

model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.5, 0.5])

model.transmat_ = np.array([[1., 0.],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1., 0.]])
model.covars_ = np.array([0.5, 0.5]) ** 2

t0 = time()
N = 5
X, Z = model.simulate_data(N, repeats = 10000)
t1 = time()
print(f"Time to simulate data: {t1 - t0:.3f}s")

f_options = {
    'mean': lambda x: np.mean(x, axis=1),
    'flatten': lambda x: x.flatten(),
}
f = f_options['mean']

I = f(X[..., 0])
Q = f(X[..., 1])

t0 = time()
model_to_fit = GaussianModel(n_components=2, covariance_type="spherical")
model_to_fit.fit(X)

t1 = time()
print(f"Time to initialize and fit model: {t1 - t0:.3f}s")

print(f"startprob:  \n{model_to_fit.startprob_}")
print(f"transmat: \n{model_to_fit.transmat_}")

print(f"means: \n{model_to_fit.means_}")
print(f"std: \n{model_to_fit.covar_to_std()}")

IQ_range = max([I.max() - I.min(), Q.max() - Q.min()])
I_bins = np.linspace(I.mean() -IQ_range / 2, I.mean() + IQ_range / 2, 40)
Q_bins = np.linspace(Q.mean() -IQ_range / 2, Q.mean() + IQ_range / 2, 40)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.set_size_inches(5, 2.5)
ax[0].scatter(I, Q, s = 1)
ax[0].hist2d(I, Q, cmap='hot', bins = [I_bins, Q_bins])
# ax[0].scatter(model_to_fit.means_[:, 0], model_to_fit.means_[:, 1], c='b', s=10)

for i in range(model_to_fit.n_components):
    std = model_to_fit.covar_to_std()[i] / np.sqrt(N)
    ax[0].add_artist(plt.Circle(model_to_fit.means_[i], 2 * std, color='b', fill=False))
ax[0].set_xlabel('I')
ax[0].set_ylabel('Q')
ax[0].set_xlim(I_bins[0], I_bins[-1])
ax[0].set_ylim(Q_bins[0], Q_bins[-1])

ax[1].hist(I, bins=40, density=True, alpha=0.5, label='I')
plt.show()
