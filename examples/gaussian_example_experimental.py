import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from time import time

from errorcausation import GaussianModel
from scipy.constants import h, Boltzmann

import scienceplots


plt.style.use(['science', 'no-latex', 'grid', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

data = np.load('./data/superconducting/sequential_data_oxford.npz')
I, Q = data.get('I'), data.get('Q')
X = np.stack([I, Q], axis=-1)
X_std = X.std()

N = 10000
X = X[:N, ...]
I = I[:N, ...]
Q = Q[:N, ...]

I /= X_std
Q /= X_std
X /= X_std

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
I_bins = np.linspace(I.mean() -IQ_range / 2, I.mean() + IQ_range / 2, 100)
Q_bins = np.linspace(Q.mean() -IQ_range / 2, Q.mean() + IQ_range / 2, 100)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=False, sharex=False, dpi=300)
fig.set_size_inches(5, 2.5)
ax[0, 0].imshow(I.T, cmap='hot', aspect='auto', origin='lower', extent=[0, I.shape[0], 0, I.shape[1]], interpolation='antialiased')
ax[0, 0].set_title('I')
ax[0, 0].set_ylabel('Repeat')
ax[0, 0].set_ylabel('Measurement')

ax[0, 1].imshow(Q.T, cmap='hot', aspect='auto', origin='lower', extent=[0, Q.shape[0], 0, Q.shape[1]], interpolation='antialiased')
ax[0, 1].set_title('Q')
ax[0, 1].set_ylabel('Repeat')

ax[1, 0].hist2d(I.flatten(), Q.flatten(), bins=[I_bins, Q_bins], cmap='hot', density=True, rasterized=True)
ax[1, 0].scatter(model_to_fit.means_[:, 0], model_to_fit.means_[:, 1], c='b', s=10)
ax[1, 0].set_xlabel('I')
ax[1, 0].set_ylabel('Q')
ax[1, 0].set_aspect('equal', 'box')

for i in range(2):
    std = model_to_fit.covar_to_std()
    circle = plt.Circle(model_to_fit.means_[i, :], 2 * std[i], fill=False, color='b')
    ax[1, 0].add_artist(circle)

ax[1, 1].scatter(I.flatten(), Q.flatten(), s=0.1, c = 'k', alpha=0.5, marker = '.')
ax[1, 1].scatter(model_to_fit.means_[:, 0], model_to_fit.means_[:, 1], c='b', s=10,  marker = 'o')
ax[1, 1].set_xlabel('I')
ax[1, 1].set_ylabel('Q')


ax[1, 1].set_xlim(I_bins[0], I_bins[-1])
ax[1, 1].set_ylim(Q_bins[0], Q_bins[-1])
ax[1, 1].set_aspect('equal', 'box')

for i in range(2):
    std = model_to_fit.covar_to_std()
    circle = plt.Circle(model_to_fit.means_[i, :], 2 * std[i], fill=False, color='b')
    ax[1, 1].add_artist(circle)

plt.savefig('./data/superconducting/sequential_data.pdf')


delta_E = 5e9 * h
r = model_to_fit.startprob_.min() / model_to_fit.startprob_.max()
T = - delta_E / (Boltzmann * np.log(r))



print(f'The fridge temperature is {1000 * T:.1f}mK')
print(f'The initialisation fidelity is {100 * model_to_fit.startprob_.max():.1f}%')
print(f'The model\'s stead-state indicates a initialisation fidelity of {100 * model_to_fit.get_stationary_distribution().max():.1f}%')

model_to_fit.predict(X)