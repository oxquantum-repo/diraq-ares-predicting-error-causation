from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

from .helper_functions import *



class Model(hmm.CategoricalHMM):
	
	def __init__(self):
		super(Model, self).__init__(n_components=2, init_params='')
	
	def __repr__(self):
		P_init_even, P_spin_flip_even_to_odd, P_spin_flip_odd_to_even, P_readout = self.get_probabilities()
		return f"P_init_even: {P_init_even :.3f}\n" \
			   f"P_spin_flip_even_to_odd: {P_spin_flip_even_to_odd :.3f}\n" \
			   f"P_spin_flip_odd_to_even: {P_spin_flip_odd_to_even :.3f}\n" \
			   f"P_readout: {P_readout :.3f}"
		
	def set_probabilities(self, P_init_even: float, P_spin_flip_even_to_odd: float,
						  P_spin_flip_odd_to_even, P_readout: float, tol = 1e-3):
		
		P_init_even = np.clip(P_init_even, tol, 1 - tol)
		P_spin_flip_even_to_odd = np.clip(P_spin_flip_even_to_odd, tol, 1 - tol)
		P_spin_flip_odd_to_even = np.clip(P_spin_flip_odd_to_even, tol, 1 - tol)
		P_readout = np.clip(P_readout, tol, 1 - tol)
		
		# the probability of initialising in the two states [even, odd]
		self.startprob_ = np.array([P_init_even, 1 - P_init_even])
		
		# the transition matrix [[even-even, even-odd], [odd-even, odd-odd]]
		self.transmat_ = np.array([
			[1 - P_spin_flip_even_to_odd, P_spin_flip_even_to_odd],
			[P_spin_flip_odd_to_even, 1 - P_spin_flip_odd_to_even]
		])
		
		# the emission matrix encoding the readout fidelity
		# [[readout even when even, readout even when odd],
		# [readout odd when even, readout odd when odd]]
		self.emissionprob_ = np.array([
			[P_readout, 1 - P_readout],
			[1 - P_readout, P_readout]
		])
		return self
	
	def get_probabilities(self):
		# taking the parameters out of matrix form
		P_init_even = self.startprob_[0]
		P_spin_flip_even_to_odd = self.transmat_[0, 1]
		P_spin_flip_odd_to_even = self.transmat_[1, 0]
		P_readout = (self.emissionprob_[0, 0] + self.emissionprob_[1, 1]) / 2
		return P_init_even, P_spin_flip_even_to_odd, P_spin_flip_odd_to_even, P_readout
	
	def randomly_set_probabilities(self, P_init_even_prior: float, P_spin_flip_even_to_odd_prior: float,
								   P_spin_flip_odd_to_even_prior,
								   P_readout_prior: float, std=0.01):
		
		P_init_even = beta_distribution_from_mean_and_std(P_init_even_prior, std)
		P_spin_flip_even_to_odd = beta_distribution_from_mean_and_std(P_spin_flip_even_to_odd_prior, std)
		P_spin_flip_odd_to_even = beta_distribution_from_mean_and_std(P_spin_flip_odd_to_even_prior, std)
		P_readout = beta_distribution_from_mean_and_std(P_readout_prior, std)
		return self.set_probabilities(P_init_even, P_spin_flip_even_to_odd, P_spin_flip_odd_to_even, P_readout)
	
	def fit(self, X):
		lengths = np.full(X.shape[0], fill_value=X.shape[1])
		return super().fit(X.reshape(-1, 1), lengths)
	
	def score(self, X, lengths=None):
		lengths = np.full(X.shape[0], fill_value=X.shape[1])
		return super().score(X.reshape(-1, 1), lengths)
	
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
		measured_states = np.array(measured_states).squeeze()
		true_states = np.array(true_states).squeeze()
		
		if plot:
			fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
			
			# plotting the generated data
			ax[0].imshow(true_states.T,
						 cmap='Greys', aspect='auto',
						 origin='lower', interpolation='none'
						 )
			ax[0].set_title("true_states")
			ax[0].set_xlabel('# repeat')
			ax[0].set_ylabel('# measurement')
			
			# plotting the generated data
			ax[1].set_title("measured_states")
			ax[1].imshow(measured_states.T,
						 cmap='Greys', aspect='auto',
						 origin='lower', interpolation='none'
						 )
			ax[1].set_xlabel('# repeat')
			
			# plotting the generated data
			ax[2].set_title("|true - measured|")
			ax[2].imshow(np.abs(measured_states.T - true_states.T),
						 cmap='Greys', aspect='auto',
						 origin='lower', interpolation='none'
						 )
			ax[2].set_xlabel('# repeat')
			plt.show()
		
		return measured_states, true_states
