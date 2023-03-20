import matplotlib.pyplot as plt

from .priors import calculate_priors
from tqdm import tqdm
from .model import Model
import numpy as np
from numdifftools import Hessian

def fit_models(measured_states, priors, priors_std, number_of_models_to_fit = 10, hessian_step = 1e-6, plot = False):
	models_to_fit = [Model() for _ in range(number_of_models_to_fit)]
	
	for model in tqdm(models_to_fit):
		model.randomly_set_probabilities(*priors, *priors_std)
		model.fit(measured_states)
	
	best_model = max(models_to_fit, key=lambda model: model.score(measured_states))
	best_models_parameters = np.array(best_model.get_probabilities())

	def f(x):
		assert not np.any(x < 0), f"trying to set probability less than zero {x}"
		assert not np.any(x > 1), f"trying to set probability greater than one {x}"
		return Model().set_probabilities(*x).score(measured_states)
	
	I = - Hessian(f, step=hessian_step, method='backward').__call__(best_models_parameters)
	I_inv = np.linalg.pinv(I)
	parameter_errors = np.sqrt(np.diag(I_inv))


	if plot:
		log = False


		parameters = np.array([model.get_probabilities() for model in models_to_fit])
		fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
		fig.suptitle('Model fits', fontsize=16)

		ax[0, 0].hist(parameters[:, 0], bins=number_of_models_to_fit // 10, log=log)
		ax[0, 0].axvline(best_models_parameters[0], linestyle='-.', c='k')
		# ax[0, 0].set_xlim(0, 1)

		ax[0, 1].hist(parameters[:, 1], bins=number_of_models_to_fit // 10, log=log)
		ax[0, 1].axvline(best_models_parameters[1], linestyle='-.', c='k')
		# ax[0, 1].set_xlim(0, 1)

		ax[1, 0].hist(parameters[:, 2], bins= number_of_models_to_fit // 10, log=log)
		ax[1, 0].axvline(best_models_parameters[2], linestyle='-.', c='k')
		# ax[1, 0].set_xlim(0, 1)

		ax[1, 1].hist(parameters[:, 3], bins=number_of_models_to_fit // 10, log=log)
		ax[1, 1].axvline(best_models_parameters[3], linestyle='-.', c='k')
		# ax[1, 1].set_xlim(0, 1)


	return best_models_parameters, parameter_errors


