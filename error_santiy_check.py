import matplotlib.pyplot as plt

from src import Model, calculate_priors, fit_models
from tqdm import tqdm

import numpy as np
from numdifftools import Hessian

number_of_models_to_fit = 10000
hessian_step=1e-6
plotting_parameter_window = 0.2

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

true_model = Model()
true_probabilities = np.array([0.95, 0.02, 0.02, 0.98])
true_model.set_probabilities(*true_probabilities)
priors_std = [0.05, 0.05, 0.05, 0.05]


models_to_fit = [Model(tol = 0.001, n_iter = 1000) for _ in range(number_of_models_to_fit)]

for model in tqdm(models_to_fit):
    measured_states, true_states = true_model.simulate_data(20, 100, plot=False)

    priors = calculate_priors(measured_states)

    model.randomly_set_probabilities(*priors, *priors_std)
    model.fit(measured_states)

def f(x):
    assert not np.any(x < 0), f"trying to set probability less than zero {x}"
    assert not np.any(x > 1), f"trying to set probability greater than one {x}"
    return Model().set_probabilities(*x).score(measured_states)

I = - Hessian(f, step=hessian_step, method='backward').__call__(true_probabilities)
I_inv = np.linalg.pinv(I)
parameter_errors = np.sqrt(np.diag(I_inv))

plot = True
if plot:
    log = False

    parameters = np.array([model.get_probabilities() for model in models_to_fit])
    if plotting_parameter_window is not None:
        parameter_differences = parameters - true_probabilities[np.newaxis, :]
        parameters = parameters[np.all(np.abs(parameter_differences) < plotting_parameter_window, axis=1), :]

    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.suptitle('Model fits', fontsize=16)

    ax[0, 0].hist(parameters[:, 0], bins=number_of_models_to_fit // 10, log=log)
    ax[0, 0].axvline(true_probabilities[0] - parameter_errors[0], linestyle='-.', c='k')
    ax[0, 0].axvline(true_probabilities[0] + parameter_errors[0], linestyle='-.', c='k')
    ax[0, 0].axvline(true_probabilities[0], linestyle='-.', c='k')


    ax[0, 1].hist(parameters[:, 1], bins=number_of_models_to_fit // 10, log=log)
    ax[0, 1].axvline(true_probabilities[1], linestyle='-.', c='k')
    ax[0, 1].axvline(true_probabilities[1] - parameter_errors[1], linestyle='-.', c='k')
    ax[0, 1].axvline(true_probabilities[1] + parameter_errors[1], linestyle='-.', c='k')
    ax[0, 1].axvline(true_probabilities[1], linestyle='-.', c='k')


    ax[1, 0].hist(parameters[:, 2], bins=number_of_models_to_fit // 10, log=log)
    ax[1, 0].axvline(true_probabilities[2], linestyle='-.', c='k')
    ax[1, 0].axvline(true_probabilities[2] - parameter_errors[2], linestyle='-.', c='k')
    ax[1, 0].axvline(true_probabilities[2] + parameter_errors[2], linestyle='-.', c='k')
    ax[1, 0].axvline(true_probabilities[2], linestyle='-.', c='k')


    ax[1, 1].hist(parameters[:, 3], bins=number_of_models_to_fit // 10, log=log)
    ax[1, 1].axvline(true_probabilities[3], linestyle='-.', c='k')
    ax[1, 1].axvline(true_probabilities[3] - parameter_errors[3], linestyle='-.', c='k')
    ax[1, 1].axvline(true_probabilities[3] + parameter_errors[3], linestyle='-.', c='k')
    ax[1, 1].axvline(true_probabilities[3], linestyle='-.', c='k')


