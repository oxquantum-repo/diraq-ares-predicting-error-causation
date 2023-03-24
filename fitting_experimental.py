from src import Model, calculate_priors, fit_models
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

data_files = {
    'even': '/Users/barnaby/Documents/charred_qubits/diraq_hmm/data/experimental data/Repeated_readout_1000_measurements_20_repeats_run_even_init_18433.mat',
    'odd': '/Users/barnaby/Documents/charred_qubits/diraq_hmm/data/experimental data/Repeated_readout_1000_measurements_20_repeats_run_odd_init_18463.mat',
    'superpostion': '/Users/barnaby/Documents/charred_qubits/diraq_hmm/data/experimental data/Repeated_readout_1000_measurements_20_repeats_run_superposition_init_18450.mat',
    'repeats': '/Users/barnaby/Documents/charred_qubits/diraq_hmm/data/experimental data/Repeated_readout_10000_measurements_20_repeats_run_19071.mat'
}

name = 'odd'
data = loadmat(data_files.get(name))
repeat = data['repeats'].squeeze()
measurement = data['measurements'].squeeze()
measured_states = 1 - data['measured_states'].squeeze()[:100, :]

priors = [0.05, 0.02, 0.02, 0.99, 0.90]
priors_std = [0.01, 0.01, 0.01, 0.01, 0.01]

best_models_parameters, errors, model = fit_models(measured_states, priors, priors_std, number_of_models_to_fit=1,
                                            plot=True, plotting_parameter_window=0.5)
best_fitting_model = Model().set_probabilities(*best_models_parameters)

predicted_true_states = best_fitting_model.predict(measured_states)
np.savez(f'./{name}.npz', measured_states=measured_states, predicted_true_states=predicted_true_states)

names = ['p_init_even', 'p_even_to_odd', 'p_odd_to_even', 'p_readout_even', 'p_readout_odd']
for name, value, error in zip(names, best_models_parameters, errors):
    print(f"{name} = {value:.4f} +/- {error:.4f}")


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


    print(f"Number of even readout errors: {np.sum(diff == 1)}")
    print(f"Number of odd readout errors: {np.sum(diff == -1)}")

    print(f"fraction of even readout errors: {1 - np.sum(diff == 1) / np.sum(predicted_true_states ==0) :.4f}")
    print(f"fraction of odd readout errors: {1 - np.sum(diff == -1) / np.sum(predicted_true_states == 1) :.4f}")