from src import CategoricalModel, fit_models
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
from pathlib import Path

import scienceplots

plt.style.use(['science', 'no-latex', 'grid', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

priors = [0.05, 0.02, 0.02, 0.99, 0.90]
priors_std = [0.01, 0.01, 0.01, 0.01, 0.01]

data_name = 'even'
data_folder = Path("./data/")

data_files = {
    'even': data_folder / 'Repeated_readout_1000_measurements_20_repeats_run_even_init_18433.mat',
    'odd': data_folder / 'Repeated_readout_1000_measurements_20_repeats_run_odd_init_18463.mat',
    'superposition': data_folder / 'Repeated_readout_1000_measurements_20_repeats_run_superposition_init_18450.mat',
}

file = Path(data_files.get(data_name))

data = loadmat(file)
repeat = data['repeats'].squeeze()
measurement = data['measurements'].squeeze()
measured_states = 1 - data['measured_states'].squeeze()[0:200]

shape = np.array(measured_states.shape) + 1
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.set_size_inches(7, 2.5)

for file, ax in zip(['even', 'odd', 'superposition'], axs):
    data = loadmat(data_files.get(file))
    measured_states = 1 - data['measured_states'].squeeze()[0:200]
    ax.imshow(measured_states, cmap=ListedColormap(['white', 'black']), aspect='auto', origin='lower',
              interpolation='antialiased', extent=[1, shape[1], 1, shape[0]])
    ax.set_xlabel('Measurement\nnumber')
    ax.set_title(file.capitalize())

axs[0].set_ylabel('Repeat')

for a, label, offset in zip(axs, 'abcdefghijklmnop', [-0.1, 0, 0]):
    a.text(offset, 1.1, f'({label})', transform=a.transAxes, fontweight='bold', va='top', ha='right')
plt.savefig('/Users/barnaby/Documents/thesis/thesis/chapter9/figures/diraq_experimental_data.pdf', bbox_inches='tight')
