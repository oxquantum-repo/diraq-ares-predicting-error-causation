from src import Model, calculate_priors
import numpy as np
from tqdm import tqdm
from numdifftools import Hessian


true_model = Model()
true_model.set_probabilities(0.9, 0.02, 0.05, 0.99)
measured_states, true_states = true_model.simulate_data(20, 100)

priors = calculate_priors(measured_states)
models_to_fit = [Model() for _ in range(10)]

for model in tqdm(models_to_fit):
	model.randomly_set_probabilities(*priors, std=0.05)
	model.fit(measured_states)

best_model = max(models_to_fit, key=lambda model: model.score(measured_states))
best_models_parameters = np.array(best_model.get_probabilities())

def f(x):
	assert not np.any(x < 0), f"trying to set probability less than zero {x}"
	assert not np.any(x > 1), f"trying to set probability greater than one {x}"
	return Model().set_probabilities(*x).score(measured_states)

I = - Hessian(f, step=1e-6).__call__(best_models_parameters)
I_inv = np.linalg.pinv(I)
errors = np.sqrt(np.diag(I_inv))


true_parameters = np.array(true_model.get_probabilities())

print(true_parameters)
print(best_models_parameters)
print(errors)
print(np.abs(true_parameters - best_models_parameters))
print(np.abs(true_parameters - best_models_parameters) < 2 * errors)