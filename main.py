from src import Model, calculate_priors, fit_models
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

true_model = Model()
true_model.set_probabilities(0.90, 0.02, 0.01, 0.98)
measured_states, true_states = true_model.simulate_data(20, 1000, plot=False)

priors = calculate_priors(measured_states)
priors_std = [0.05, 0.05, 0.05, 0.05]

best_models_parameters, errors = fit_models(measured_states, priors, priors_std, number_of_models_to_fit = 10)
best_fitting_model = Model().set_probabilities(*best_models_parameters)

true_parameters = np.array(true_model.get_probabilities())


print(f"True parameters                     :{true_parameters}")
print(f"Fitted parameters                   :{best_models_parameters}")
print(f"Fitted parameter errors             :{errors}")
print(f"Fitted parameters - True_parameters :{best_models_parameters - true_parameters}")
print(f"True value within 2 std             :{np.abs(true_parameters - best_models_parameters) < 2 * errors}")


predicted_true_states = best_fitting_model.predict(measured_states)

plot_prediction = True
if plot_prediction:
	fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, constrained_layout=True)
	fig.suptitle('Model Predictions', fontsize=16)
	# plotting the generated data
	ax[0, 0].imshow(true_states.T,
				 cmap='Greys', aspect='auto',
				 origin='lower', interpolation='none'
				 )
	ax[0, 0].set_title("true_states")
	ax[0, 0].set_xlabel('# repeat')
	ax[0, 0].set_ylabel('# measurement')
	
	ax[0, 1].imshow(measured_states.T,
					cmap='Greys', aspect='auto',
					origin='lower', interpolation='none'
					)
	ax[0, 1].set_title("measured_states")
	ax[0, 1].set_xlabel('# repeat')
	ax[0, 1].set_ylabel('# measurement')
	
	# plotting the generated data
	ax[1, 0].set_title("predicted_true_states")
	ax[1, 0].imshow(predicted_true_states.T,
				 cmap='Greys', aspect='auto',
				 origin='lower', interpolation='none'
				 )
	ax[1, 0].set_xlabel('# repeat')
	
	# plotting the generated data
	ax[1, 1].set_title("|true - predicted|")
	ax[1, 1].imshow(np.abs(predicted_true_states.T - true_states.T),
				 cmap='Greys', aspect='auto',
				 origin='lower', interpolation='none'
				 )
	ax[1, 1].set_xlabel('# repeat')
	plt.show()