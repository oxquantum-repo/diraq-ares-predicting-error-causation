from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

from errorcausation.helperfunctions.arraymanipulations import full_covariance_matrix_to_spherical

class GaussianModel(hmm.GaussianHMM):

    def __init__(self, **kwargs):
        super(GaussianModel, self).__init__(**kwargs)

    def covar_to_std(self):
        return np.sqrt(np.diag(self._covars_))

    def fit(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        reshaped_data = X.reshape(-1, 2)
        return super().fit(reshaped_data, lengths)

    def predict(self, X):
        shape = X.shape
        lengths = np.full(shape[0], fill_value=shape[1])
        return super().predict(X.reshape(-1, 2), lengths).reshape(*shape)

    def score(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score(X.reshape(-1, 2), lengths)

    def score_samples(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score_samples(X.reshape(-1, 2), lengths)

    def simulate_data(self, measurements, repeats, plot=False):
        measured_states = []  # array to hold the measured states (even or odd), this data is available
        true_states = []  # array to hold the true states (even or odd), this data is hidden
        # generating the data
        for i in range(repeats):
            measured_state, true_state = self.sample(measurements)
            # appending the data to the arrays
            measured_states.append(measured_state)
            true_states.append(true_state.squeeze())
        # making the data arrays (python lists) into numpy arrays for convenience
        measured_states = np.array(measured_states)
        true_states = np.array(true_states)

        return measured_states, true_states
