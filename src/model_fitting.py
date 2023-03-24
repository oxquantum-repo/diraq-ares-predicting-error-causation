import matplotlib.pyplot as plt

from .priors import calculate_priors
from tqdm import tqdm
from .model import Model
import numpy as np
from numdifftools import Hessian

def calculate_uncertainty(measured_states, parameters, hessian_step = 1e-6):
    def f(x):
        assert not np.any(x < 0), f"trying to set probability less than zero {x}"
        assert not np.any(x > 1), f"trying to set probability greater than one {x}"
        return Model().set_probabilities(*x).score(measured_states)

    I = - Hessian(f, step=hessian_step, method='backward').__call__(parameters)
    I_inv = np.linalg.pinv(I)
    return np.sqrt(np.diag(I_inv))


def fit_models(measured_states, priors, priors_std, number_of_models_to_fit=10, plot=False,
               plotting_parameter_window=None):
    models_to_fit = [Model(tol = 0.001, n_iter = 1000) for _ in range(number_of_models_to_fit)]

    for model in tqdm(models_to_fit):
        model.randomly_set_probabilities(*priors, *priors_std)
        model.fit(measured_states)

    best_model = max(models_to_fit, key=lambda model: model.score(measured_states))
    best_models_parameters = np.array(best_model.get_probabilities())

    parameter_errors = calculate_uncertainty(measured_states, best_models_parameters)

    if plot:
        log = False
        parameters = np.array([model.get_probabilities() for model in models_to_fit])
        if plotting_parameter_window is not None:
            parameter_differences = parameters - best_models_parameters[np.newaxis, :]
            parameters = parameters[np.all(np.abs(parameter_differences) < plotting_parameter_window, axis=1), :]

        fig, ax = plt.subplots(nrows=1, ncols=5, constrained_layout=True)
        fig.suptitle('Model fits', fontsize=16)

        for i in range(5):
            number_of_bins = max(5, number_of_models_to_fit // 10)
            ax[i].hist(parameters[:, i], bins=number_of_bins, log=log)
            ax[i].axvline(best_models_parameters[i], color='r')

    return best_models_parameters, parameter_errors, best_model
