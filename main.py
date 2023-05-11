from src import CatagoricalModel, fit_models
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

true_model = CatagoricalModel()
true_probabilities = [0.0574, 0.0373, 0.0091, 0.99, 0.90]

true_model.set_probabilities(*true_probabilities)
measured_states, true_states = true_model.simulate_data(20, 100, plot=False)

priors = [0.05, 0.01, 0.01, 0.9, 0.9]
priors_std = [0.01, 0.01, 0.01, 0.01, 0.01]

best_models_parameters, errors, best_model = fit_models(measured_states, priors, priors_std, number_of_models_to_fit = 10, plot=True, plotting_parameter_window =0.5)
best_fitting_model = CatagoricalModel().set_probabilities(*best_models_parameters)

true_parameters = np.array(true_model.get_probabilities())


names = ['p_init_even', 'p_even_to_odd', 'p_odd_to_even', 'p_readout_even', 'p_readout_odd']
for i, (name, value, error) in enumerate(zip(names, best_models_parameters, errors)):
    print(f"{name} = {value:.4f} +/- {error:.4f} -- true value = {true_parameters[i]:.4f} -- within 2 std {np.abs(value - true_parameters[i]) < 2*error}")

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