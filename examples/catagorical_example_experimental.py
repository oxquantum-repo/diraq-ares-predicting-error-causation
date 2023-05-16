from src import CategoricalModel
import numpy as np
from pathlib import Path
from scipy.io import loadmat

np.random.seed(0)

file = Path('./data/Repeated_readout_1000_measurements_20_repeats_run_even_init_18433.mat')
data = loadmat(file.resolve())
measured_states = 1 - data['measured_states'].squeeze()

# initialising a model to fit to the data and setting the starting guess of parameters for the Baum-Welch algorithm
# to optimise
model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.95)
model_to_fit.set_transition_prob(0.05, 0.05)
model_to_fit.set_emission_prob(0.95, 0.95)

# fitting the model to the data, using the Baum-Welch algorithm. The uncertainty in the parameters is also computed
# using the Cramer-Rao lower bound.
model_to_fit.fit(measured_states, compute_uncertainty=True)

# printing the fitted model, which should be close to the model used to simulate the data
print(model_to_fit)

# using the fitted model to predict the true qubit state from the measured state and plotting the results
predicted_states = model_to_fit.predict(measured_states, plot=True)
