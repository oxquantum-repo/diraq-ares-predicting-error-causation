from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

from .helper_functions import *
from numdifftools import Hessian
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 10})
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class CategoricalModel(hmm.CategoricalHMM):

    def __init__(self, **kwargs):
        super(CategoricalModel, self).__init__(n_components=2, init_params='', **kwargs)
        self._errors = None

    def __repr__(self):

        p_init = self.get_start_prob()
        p_even_odd, p_odd_even = self.get_transition_prob()
        p_readout_even, p_readout_odd = self.get_emission_prob()
        if self._errors is not None:
            p_init_error = self._errors[0]
            p_even_odd_error, p_odd_even_error = self._errors[1:3]
            p_readout_even_error, p_readout_odd_error = self._errors[3:5]
            return f"p_init_even: {p_init :.3f} ± {p_init_error :.3f}\n" \
                   f"p_even_odd, p_odd_even: {p_even_odd :.3f} ± {p_even_odd_error :.3f}, {p_odd_even: 3f} ± {p_odd_even_error :.3f}\n" \
                   f"p_readout_even, p_readout_odd: {p_readout_even :.3f} ± {p_readout_even_error :.3f}, {p_readout_odd :.3f} ± {p_readout_odd_error :.3f}"
        else:
            return f"p_init_even: {p_init :.3f}\n" \
                   f"p_even_odd, p_odd_even: {p_even_odd :.3f}{p_odd_even: 3f}\n" \
                   f"p_readout_even, p_readout_odd: {p_readout_even :.3f}, {p_readout_odd :.3f}"

    def set_start_prob(self, p_init_even: float):
        self.startprob_ = np.array([p_init_even, 1 - p_init_even])
        return self

    def set_start_prob_from_beta(self, mean, std):
        p_init_even = np.random.beta(mean * (1 - std), (1 - mean) * (1 - std))
        self.set_start_prob(p_init_even)
        return self

    def get_start_prob(self):
        assert np.isclose(np.sum(self.startprob_), 1.), "The initialisation probabilities do not sum to 1"
        return self.startprob_[0]

    def get_start_prob_and_error(self):
        assert self._errors is not None, "The errors have not been calculated yet"
        return self.startprob_[0], self._errors[0]

    def set_transition_prob(self, p_spin_flip_even_to_odd: float, p_spin_flip_odd_to_even: float):
        self.transmat_ = np.array([
            [1 - p_spin_flip_even_to_odd, p_spin_flip_even_to_odd],
            [p_spin_flip_odd_to_even, 1 - p_spin_flip_odd_to_even]
        ])
        return self

    def get_transition_prob(self):
        assert np.isclose(np.sum(self.transmat_, axis=1), 1.).all(), "The transition probabilities do not sum to 1"
        return self.transmat_[0, 1], self.transmat_[1, 0]

    def get_transition_prob_and_error(self):
        assert self._errors is not None, "The errors have not been calculated yet"
        return self.transmat_[0, 1], self.transmat_[1, 0], self._errors[1], self._errors[2]

    def set_emission_prob(self, p_readout_even: float, p_readout_odd: float):
        self.emissionprob_ = np.array([
            [p_readout_even, 1 - p_readout_even],
            [1 - p_readout_odd, p_readout_odd]
        ])
        return self

    def get_emission_prob(self):
        assert np.isclose(np.sum(self.emissionprob_, axis=1), 1.).all(), "The emission probabilities do not sum to 1"
        return self.emissionprob_[0, 0], self.emissionprob_[1, 1]

    def get_emission_prob_and_error(self):
        assert self._errors is not None, "The errors have not been calculated yet"
        return self.emissionprob_[0, 0], self.emissionprob_[1, 1], self._errors[3], self._errors[4]

    def set_probs(self, p_init_even: float, p_spin_flip_even_to_odd: float,
                  p_spin_flip_odd_to_even, p_readout_even: float, p_readout_odd: float):

        self.set_start_prob(p_init_even)
        self.set_transition_prob(p_spin_flip_even_to_odd, p_spin_flip_odd_to_even)
        self.set_emission_prob(p_readout_even, p_readout_odd)
        return self

    def get_probs(self):
        return (self.get_start_prob(), *self.get_transition_prob(), *self.get_emission_prob())

    def get_probs_and_errors(self):
        return self.get_probs(), tuple(self._errors)

    def fit(self, X, compute_uncertainty=True):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        super().fit(X.reshape(-1, 1), lengths)
        if compute_uncertainty:
            self.compute_uncertainty(X)
        return self

    def compute_uncertainty(self, X, hessian_step=1e-4):
        def f(x):
            assert not np.any(x < 0), f"trying to set probability less than zero {x}"
            assert not np.any(x > 1), f"trying to set probability greater than one {x}"
            return CategoricalModel().set_probs(*x).score(X)

        I = - Hessian(f, step=hessian_step, method='backward').__call__(self.get_probs())
        I_inv = np.linalg.pinv(I)
        diag = np.diag(I_inv)
        with np.errstate(invalid='ignore'):
            if np.any(diag < 0):
                print(f"Warning: negative diagonal elements in the covariance matrix {diag}")
                self._errors = np.zeros_like(diag)
            self._errors = np.sqrt(diag)
        return self

    def predict(self, X):
        shape = X.shape
        lengths = np.full(shape[0], fill_value=shape[1])
        return super().predict(X.reshape(-1, 1), lengths).reshape(*shape)

    def score(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score(X.reshape(-1, 1), lengths)

    def score_samples(self, X):
        lengths = np.full(X.shape[0], fill_value=X.shape[1])
        return super().score_samples(X.reshape(-1, 1), lengths)

    def simulate_data(self, measurements, repeats, plot=False, **kwargs):
        measured_states = []  # array to hold the measured states (even or odd), this data is available
        true_states = []  # array to hold the true states (even or odd), this data is hidden
        # generating the data
        for i in range(repeats):
            measured_state, true_state = self.sample(measurements)
            # appending the data to the arrays
            measured_states.append(measured_state)
            true_states.append(true_state.squeeze())
        # making the data arrays (python lists) into numpy arrays for convenience
        measured_states = np.array(measured_states).squeeze()
        true_states = np.array(true_states).squeeze()

        if plot:
            shape = np.array(measured_states.shape) + 1
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            fig.set_size_inches(5, 2.5)

            kwargs = {
                'aspect': 'auto',
                'origin': 'lower',
                'interpolation': 'antialiased',
                'extent': [1, shape[1], 1, shape[0]]
            }

            diff = (measured_states - true_states)
            ax[0].imshow(measured_states, cmap=ListedColormap(['white', 'black']), **kwargs)
            ax[1].imshow(true_states, cmap=ListedColormap(['white', 'black']), **kwargs)
            ax[2].imshow(diff, cmap=ListedColormap(['red', 'white', 'green']), **kwargs)
            ax[0].set_ylabel('Repeat')
            ax[0].set_xlabel('Measurement\nnumber')
            ax[1].set_xlabel('Measurement\nnumber')
            ax[2].set_xlabel('Measurement\nnumber')

            ax[0].set_title('Measured')
            ax[1].set_title('True')
            ax[2].set_title('Measured - True')

            ax[2].legend(handles=[mpatches.Patch(color='red', label='-1'), mpatches.Patch(color='green', label='+1')]),

            if 'save_fig' in kwargs:
                if kwargs['save_fig']:
                    save_folder = kwargs['save_folder'] if 'save_folder' in kwargs else './'
                    figure_name = kwargs['figure_name'] if 'figure_name' in kwargs else 'simulated.pdf'
                    plt.savefig(f'{save_folder}/{figure_name}.pdf', dpi=300, bbox_inches='tight')
            plt.tight_layout()

        return measured_states, true_states
