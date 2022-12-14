from src import Model, calculate_priors, fit_models
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

true_model = Model()
true_model.set_probabilities(0.9, 0.02, 0.05, 0.95)
measured_states, true_states = true_model.simulate_data(20, 100)

priors = calculate_priors(measured_states)
priors_std = [0.05, 0.05, 0.05, 0.05]

best_models_parameters, errors = fit_models(measured_states, priors, priors_std, number_of_models_to_fit = 10)


true_parameters = np.array(true_model.get_probabilities())


print(f"True parameters                     :{true_parameters}")
print(f"Fitted parameters                   :{best_models_parameters}")
print(f"Fitted parameter errors             :{errors}")
print(f"Fitted parameters - True_parameters :{best_models_parameters - true_parameters}")
print(f"True value within 2 std             :{np.abs(true_parameters - best_models_parameters) < 2 * errors}")