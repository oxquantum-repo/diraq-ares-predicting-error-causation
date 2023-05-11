import matplotlib.pyplot as plt

from src import CatagoricalModel, calculate_uncertainty
from tqdm import tqdm

import numpy as np


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

model = CatagoricalModel()
probabilities = np.array([0.99, 0.01, 0.01, 0.99])
model.set_probabilities(*probabilities)

N_min = 100
N_max = 200
N_step = 10
N_average = 10


errors = np.full(fill_value=np.nan, shape=(N_step, N_average, 4))

ns = np.geomspace(N_min, N_max, N_step).astype(int)
for i, n in enumerate(tqdm(ns)):
    for j in range(N_average):
        measured_states, true_states = model.simulate_data(20, n)
        errors[i, j, :] = calculate_uncertainty(measured_states, probabilities, hessian_step=1e-6)

averaged_errors = np.nanmean(errors, axis=1)
for i, label in enumerate(['P_init', 'P_even_to_odd', 'P_odd_to_even', 'P_readout']):
    plt.plot(ns, averaged_errors[:, i], label = label)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()