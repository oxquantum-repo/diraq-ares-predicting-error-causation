from .priors import calculate_priors
from tqdm import tqdm
from .model import Model
import numpy as np
from numdifftools import Hessian

def fit_models(measured_states, priors, priors_std, number_of_models_to_fit = 10, hessian_step = 1e-6):
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
	return best_models_parameters, parameter_errors