from errorcausation import CategoricalModel
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# np.random.seed(0)

files = {
    'superposition_init': Path('./data/superposition_init.mat'),
    'even_init': Path('./data/even_init.mat'),
    'odd_init': Path('./data/odd_init.mat'),
}

file = files.get('even_init')
print(file.resolve())
data = loadmat(file.resolve())
measured_states = data['measured_states'].squeeze()[0:200, :]


# initialising a qm_model to fit to the data and setting the starting guess of parameters for the Baum-Welch algorithm
# to optimise
model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.5)
model_to_fit.set_transition_prob(0.05, 0.05)
model_to_fit.set_emission_prob(0.95, 0.95)

# fitting the qm_model to the data, using the Baum-Welch algorithm. The uncertainty in the parameters is also computed
# using the Cramer-Rao lower bound.
model_to_fit.fit(measured_states, compute_uncertainty=True)

# printing the fitted qm_model, which should be close to the qm_model used to simulate the data
print(model_to_fit)

# using the fitted qm_model to predict the true qubit state from the measured state and plotting the results
predicted_states = model_to_fit.predict(measured_states, plot=True)