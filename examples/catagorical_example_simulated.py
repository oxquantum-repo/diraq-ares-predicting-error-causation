import numpy as np
from errorcausation.Catagorical.categoricalmodel import CategoricalModel

np.random.seed(0)

# initialising a model to simulate data which we will fit another
model = CategoricalModel()
model.set_start_prob(0.99)
model.set_transition_prob(0.02, 0.02)
model.set_emission_prob(0.99, 0.99)

# using the model to simulate data and plotting it.
# the number of measurements is the number of measurements to perform before the qubit is reset
measured_states, true_states = model.simulate_data(measurements=20, repeats=1000, plot=False)

# initialising a model to fit to the data and setting the starting guess of parameters for the Baum-Welch algorithm
# to optimise
model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.90)
model_to_fit.set_transition_prob(0.1, 0.1)
model_to_fit.set_emission_prob(0.90, 0.90)

# fitting the model to the data, using the Baum-Welch algorithm. The uncertainty in the parameters is also computed
# using the Cramer-Rao lower bound.
model_to_fit.fit(measured_states, compute_uncertainty=True)

# printing the fitted model, which should be close to the model used to simulate the data
print(model_to_fit)

# using the fitted model to predict the true qubit state from the measured state and plotting the results
predicted_states = model_to_fit.predict(measured_states, plot=True, save_fig=True)
