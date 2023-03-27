from src import Model, calculate_priors, fit_models
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
from pathlib import Path

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

priors = [0.05, 0.02, 0.02, 0.99, 0.90]
priors_std = [0.01, 0.01, 0.01, 0.01, 0.01]

data_name = 'superposition'
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
measured_states = 1 - data['measured_states'].squeeze()


best_models_parameters, errors, model = fit_models(measured_states, priors, priors_std, number_of_models_to_fit=1, plot=False)
best_fitting_model = Model().set_probabilities(*best_models_parameters)

predicted_true_states = best_fitting_model.predict(measured_states)
np.savez(f'./{data_name}.npz', measured_states=measured_states, predicted_true_states=predicted_true_states)

names = ['p_init_even', 'p_even_to_odd', 'p_odd_to_even', 'p_readout_even', 'p_readout_odd']
for data_name, value, error in zip(names, best_models_parameters, errors):
    print(f"{data_name} = {value:.4f} +/- {error:.4f}")


plot_data = True
if plot_data:
    fig, ax = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    fig.suptitle('Model fits', fontsize=16)

    diff = (measured_states - predicted_true_states).T

    ax[0].imshow(measured_states.T, cmap=ListedColormap(['white', 'black']), aspect='auto', origin='lower')
    ax[1].imshow(predicted_true_states.T, cmap=ListedColormap(['white', 'black']), aspect='auto', origin='lower')
    ax[2].imshow((measured_states - predicted_true_states).T, cmap=ListedColormap(['red', 'white', 'green']), aspect='auto', origin='lower', interpolation='none')

    ax[0].set_ylabel('# repeated readout')
    ax[0].set_xlabel('# measurement')
    ax[1].set_xlabel('# measurement')
    ax[2].set_xlabel('# measurement')

    ax[0].set_title('Measured states')
    ax[1].set_title('Predicted true states')
    ax[2].set_title('Difference between \n measured and predicted states')
    plt.show()
