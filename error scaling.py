import matplotlib.pyplot as plt

from src import CategoricalModel, calculate_uncertainty
from tqdm import tqdm

import numpy as np

import scienceplots
plt.style.use(['science', 'no-latex', 'grid', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

model = CategoricalModel()
probabilities = np.array([0.99, 0.01, 0.01, 0.99])
model.set_probabilities(*probabilities)

N_min = 5
N_max = 2000
N_step = 10
N_average = 100


errors = np.full(fill_value=np.nan, shape=(N_step, N_average, 4))

ns = np.geomspace(N_min, N_max, N_step).astype(int)
for i, n in enumerate(tqdm(ns)):
    for j in range(N_average):
        measured_states, true_states = model.simulate_data(20, n)
        errors[i, j, :] = calculate_uncertainty(measured_states, probabilities, hessian_step=1e-6)


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 2.5)

averaged_errors = np.nanmean(errors, axis=1)
for i, label in enumerate(['P_init', 'P_even_to_odd', 'P_odd_to_even', 'P_readout']):
    ax.plot(ns, averaged_errors[:, i], label = label)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.show()