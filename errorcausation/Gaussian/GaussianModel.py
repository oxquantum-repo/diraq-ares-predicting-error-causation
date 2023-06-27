from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.interpolate import interp1d

from .helper_functions import readout_fidelity, readout_corrections

class GaussianModel(hmm.GaussianHMM):

    def __init__(self, **kwargs):
        super(GaussianModel, self).__init__(**kwargs)

    def covar_to_std(self):
        return np.sqrt(np.diag(self._covars_))

    def rescale(self, scale_factor):
        self.means_ = self.means_ * scale_factor
        self._covars_ = self._covars_ * scale_factor ** 2

    def fit(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        reshaped_data = X.reshape(-1, 2)
        return super().fit(reshaped_data, lengths)

    def predict(self, X):
        shape = X.shape
        lengths = np.full(shape[0], fill_value=shape[1])
        return super().predict(X.reshape(-1, 2), lengths).reshape(*shape[:2])

    def score(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score(X.reshape(-1, 2), lengths)

    def score_samples(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score_samples(X.reshape(-1, 2), lengths)

    def simulate_data(self, measurements, repeats, plot=True):
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

        if plot:

            if repeats > 1:

                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                fig.set_size_inches(5, 2.5)
                for i, a in enumerate(ax):
                    a.imshow(measured_states[..., i].T, origin='lower', aspect='auto', cmap='hot', interpolation='none')
                    a.set_xlabel('Repetition')
                    a.set_ylabel('Measurement')

            else:
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                fig.set_size_inches(5, 2.5)
                for i, label in enumerate(['I', 'Q']):
                    x = np.arange(measurements) + 0.5
                    y = measured_states[..., i].squeeze()

                    f = interp1d(x, y, kind='nearest', bounds_error=False, fill_value=(y[0], y[-1]))
                    x_dense = np.linspace(0, measurements - 1, 1000)
                    y_dense = f(x_dense)

                    ax[i].plot(x_dense, y_dense, color='black', linewidth=1)
                    ax[i].set_xlabel('Measurement')
                    ax[i].set_ylabel(label)



                t = np.linspace(0, 1, measurements)
                I = measured_states[..., 0].squeeze()
                Q = measured_states[..., 1].squeeze()

                ax[2].scatter(I, Q, c = cm.Greys(t), s = 1)

                I_f = interp1d(t, I, kind='linear', bounds_error=False, fill_value=(I[0], I[-1]))
                Q_f = interp1d(t, Q, kind='linear', bounds_error=False, fill_value=(Q[0], Q[-1]))

                t_dense = np.linspace(0, 1, 10 * measurements)
                ax[2].plot(I, Q, color='black', linewidth=1)
                ax[2].set_xlabel('I')
                ax[2].set_ylabel('Q')
                ax[2].set_aspect('equal')

            fig.tight_layout()
            plt.show()

        return measured_states, true_states

    def readout_corrections(self):
        return readout_corrections(self)

    def readout_fidelity(self, X, plot = False):
        return readout_fidelity(self, X, plot)
