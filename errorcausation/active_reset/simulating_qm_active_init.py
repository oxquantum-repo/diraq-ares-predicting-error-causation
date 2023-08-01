from datetime import datetime

import numpy as np
from dataclasses import dataclass

from matplotlib import pyplot as plt
from scipy.integrate import quad


def gaussian(x, mean, std):
	return np.exp(-np.power(x - mean, 2.) / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)


def last_positive_index(x):
	return (x < 0.).argmax(axis=1) - 1


def last_positive_value(x):
	return x[np.arange(x.shape[0]), last_positive_index(x)]

def last_non_nan_index(x):
	return (np.isnan(x)).argmax(axis=1) - 1




@dataclass
class QubitInit:
	p_init_0: float

	p_1_to_0: float
	p_0_to_1: float

	x_gate_fidelity: float

	threshold: float
	std: float

	def __post_init__(self):
		self.steady_state = self.steady_state()
		self.f_ground = self.ground_readout_fidelity()
		self.f_excited = self.excited_readout_fidelity()
		self.p_less_than_threshold = self.p_less_than_threshold()
		self.emissonprob = self.emissonprob()


	def start_prob(self):
		return np.array([self.p_init_0, 1 - self.p_init_0])

	def x_gate(self):
		return np.array([
			[1 - self.x_gate_fidelity, self.x_gate_fidelity],
			[self.x_gate_fidelity, 1 - self.x_gate_fidelity]
		])

	def I_gate(self):
		return np.array([
			[1 - self.p_0_to_1, self.p_0_to_1],
			[self.p_1_to_0, 1 - self.p_1_to_0]
		])

	def transmat(self):
		I_gate = self.I_gate()
		x_gate = self.x_gate()
		return np.stack([I_gate, I_gate @ x_gate], axis=0)

	def steady_state(self):
		return np.array([self.p_1_to_0 / (self.p_0_to_1 + self.p_1_to_0), self.p_0_to_1 / (self.p_0_to_1 + self.p_1_to_0)])

	def p_less_than_threshold(self):
		return quad(lambda x: gaussian(x, 0, self.std), -np.inf, self.threshold)[0]

	def ground_readout_fidelity(self):
		area_1 = quad(lambda x: gaussian(x, 0, self.std), -np.inf, self.threshold)[0]
		area_2 = quad(lambda x: gaussian(x, 1, self.std), -np.inf, self.threshold)[0]
		return area_1 / (area_1 + area_2)

	def excited_readout_fidelity(self):
		area_1 = quad(lambda x: gaussian(x, 0, self.std), self.threshold, np.inf)[0]
		area_2 = quad(lambda x: gaussian(x, 1, self.std), self.threshold, np.inf)[0]
		return area_2 / (area_1 + area_2)

	def emissonprob(self):
		f_ground = self.ground_readout_fidelity()
		f_excited = self.excited_readout_fidelity()
		return np.array([
			[f_ground, 1 - f_ground],
			[1 - f_excited, f_excited]
		])

	def plot_readout(self, show=True):
		x = np.linspace(-1, 2, 1000)
		plt.plot(x, gaussian(x, 0.0, self.std), label="0")
		plt.plot(x, gaussian(x, 1.0, self.std), label="1")
		plt.axvline(x=self.threshold, color="black", linestyle="--")
		plt.xlabel('I threshold (a.u)')
		plt.ylabel('Probability density')
		if show:
			plt.show()

	def plot_threshold(self, show = True):
		self.plot_readout(show=False)
		plt.axvline(x=self.threshold, color="black", linestyle="--")
		if show:
			plt.show()

	def simulate_qm(self, N_repeat, iteration_max):
		return repeated_calls_qm(self, N_repeat, iteration_max)

	def simulate_hhm(self, N_repeat, iteration_max):
		return repeated_calls_hmm(self, N_repeat, iteration_max)

	def simulate_hhm_gaussian(self, N_repeat, iteration_max):
		return repeated_calls_hmm_gaussian(self, N_repeat, iteration_max)

def repeated_calls_qm(model, n_repeat, iteration_max):
	hidden_states = np.zeros((n_repeat, iteration_max), dtype=int)
	observations = np.zeros((n_repeat, iteration_max), dtype=float)

	final_hidden_states = np.zeros(n_repeat, dtype=int)
	final_observations = np.zeros(n_repeat, dtype=float)

	for i in range(n_repeat):
		observations[i, :], hidden_states[i, :], final_hidden_states[i], final_observations[i] = forward_qm(model, iteration_max)

	return QubitInitFeedbackResult(
		model=model,
		hidden_states=hidden_states,
		observations=observations,
		verification_state=final_hidden_states,
		verification_measurement=final_observations
	)

def repeated_calls_hmm(model, n_repeat, iteration_max):
	hidden_states = np.zeros((n_repeat, iteration_max), dtype=int)
	observations = np.zeros((n_repeat, iteration_max), dtype=float)

	final_hidden_states = np.zeros(n_repeat, dtype=int)
	final_observations = np.zeros(n_repeat, dtype=float)

	for i in range(n_repeat):
		observations[i, :], hidden_states[i, :], final_hidden_states[i], final_observations[i] = forward_hhm(model, iteration_max)

	return QubitInitFeedbackResult(
		model=model,
		hidden_states=hidden_states,
		observations=observations,
		verification_state=final_hidden_states,
		verification_measurement=final_observations
	)

def repeated_calls_hmm_gaussian(model, n_repeat, iteration_max):
	hidden_states = np.zeros((n_repeat, iteration_max), dtype=int)
	observations = np.zeros((n_repeat, iteration_max), dtype=float)

	final_hidden_states = np.zeros(n_repeat, dtype=int)
	final_observations = np.zeros(n_repeat, dtype=float)

	for i in range(n_repeat):
		observations[i, :], hidden_states[i, :], final_hidden_states[i], final_observations[i] = forward_hhm_gaussian(model, iteration_max)

	return QubitInitFeedbackResult(
		model=model,
		hidden_states=hidden_states,
		observations=observations,
		verification_state=final_hidden_states,
		verification_measurement=final_observations
	)

def forward_qm(model: QubitInit, N):
	"""Forward algorithm for HMMs.
		O: observation sequence
		S: set of states
		Pi: initial state probabilities
		Tm: transition matrix
		Em: emission matrix
		"""

	transmat = model.transmat()

	# the hidden states are the states of the HMM and are not accessible
	hidden_states = np.full(N, fill_value=-1, dtype=int)
	# the observations are the observations of the HMM and are accessible
	observations = np.full(N, fill_value=np.nan, dtype=float)

	starting_hidden_state = np.random.choice([0, 1], p=model.start_prob())
	p_hidden = transmat[1, starting_hidden_state, :]
	hidden_states[0] = np.random.choice([0, 1], p=p_hidden)
	observations[0] = hidden_states[0] + np.random.randn() * model.std

	for i in range(1, N):
		if observations[i - 1] < model.threshold:
			break

		p_hidden = transmat[1, hidden_states[i - 1], :]
		hidden_states[i] = np.random.choice([0, 1], p=p_hidden)
		observations[i] = hidden_states[i] + np.random.randn() * model.std

	p_hidden = transmat[0, hidden_states[i - 1], :]
	final_state = np.random.choice([0, 1], p=p_hidden)
	final_observation = final_state + np.random.randn() * model.std

	return observations, hidden_states, final_state, final_observation

def forward_hhm(model: QubitInit, N):

	transmat = model.transmat()
	emmisonprob = model.emissonprob

	# the hidden states are the states of the HMM and are not accessible
	hidden_states = np.full(N, fill_value=-1, dtype=int)
	# the observations are the observations of the HMM and are accessible
	observations = np.full(N, fill_value=np.nan, dtype=float)
	classifications = np.full(N, fill_value=-1, dtype=int)

	alpha = np.full((N, 2), fill_value=np.nan, dtype=float)
	actions = np.full(N, fill_value=-1, dtype=int)

	starting_hidden_state = np.random.choice([0, 1], p=model.start_prob())

	actions[0] = 0
	p_hidden = transmat[actions[0], starting_hidden_state, :]

	hidden_states[0] = np.random.choice([0, 1], p=p_hidden)
	observations[0] = hidden_states[0] + np.random.randn() * model.std
	classifications[0] = observations[0] > model.threshold

	# calculating alpha[0, :]
	alpha[0, :] = model.start_prob() * emmisonprob[:, classifications[0]]
	alpha[0, :] /= np.sum(alpha[0, :])

	for i in range(1, N):
		if alpha[i - 1, 0] >1 - model.p_0_to_1:
			break

		actions[i] = alpha[i - 1, 0] < 0.5

		p_hidden = transmat[actions[i], hidden_states[i - 1], :]
		hidden_states[i] = np.random.choice([0, 1], p=p_hidden)
		observations[i] = hidden_states[i] + np.random.randn() * model.std
		classifications[i] = observations[i] > model.threshold

		for m in range(2):
			alpha[i, m] = np.sum(alpha[i - 1, :] * transmat[actions[i], :, m]) * emmisonprob[m, classifications[i]]
		alpha[i, :] /= np.sum(alpha[i, :])

	p_hidden = transmat[0, hidden_states[i - 1], :]
	final_state = np.random.choice([0, 1], p=p_hidden)
	final_observation = final_state + np.random.randn() * model.std

	return observations, hidden_states, final_state, final_observation

def forward_hhm_gaussian(model: QubitInit, N):

	transmat = model.transmat()

	std = model.std

	# the hidden states are the states of the HMM and are not accessible
	hidden_states = np.full(N, fill_value=-1, dtype=int)
	# the observations are the observations of the HMM and are accessible
	observations = np.full(N, fill_value=np.nan, dtype=float)
	classifications = np.full(N, fill_value=-1, dtype=int)

	alpha = np.full((N, 2), fill_value=np.nan, dtype=float)
	actions = np.full(N, fill_value=-1, dtype=int)

	starting_hidden_state = np.random.choice([0, 1], p=model.start_prob())

	actions[0] = 0
	p_hidden = transmat[actions[0], starting_hidden_state, :]

	hidden_states[0] = np.random.choice([0, 1], p=p_hidden)
	observations[0] = hidden_states[0] + np.random.randn() * model.std
	classifications[0] = observations[0] > model.threshold

	# calculating alpha[0, :]
	alpha[0, :] = model.start_prob() * gaussian(observations[0], np.array([0, 1]), std)
	alpha[0, :] /= np.sum(alpha[0, :])

	for i in range(1, N):
		if alpha[i - 1, 0] > 1 - model.p_0_to_1:
			break

		actions[i] = alpha[i - 1, 0] < 0.5

		p_hidden = transmat[actions[i], hidden_states[i - 1], :]
		hidden_states[i] = np.random.choice([0, 1], p=p_hidden)
		observations[i] = hidden_states[i] + np.random.randn() * model.std
		classifications[i] = observations[i] > model.threshold

		for m in range(2):
			alpha[i, m] = np.sum(alpha[i - 1, :] * transmat[actions[i], :, m]) * gaussian(observations[i], m, std)
		alpha[i, :] /= np.sum(alpha[i, :])

	p_hidden = transmat[0, hidden_states[i - 1], :]
	final_state = np.random.choice([0, 1], p=p_hidden)
	final_observation = final_state + np.random.randn() * model.std

	return observations, hidden_states, final_state, final_observation


@dataclass
class QubitInitFeedbackResult:
	model: QubitInit
	hidden_states: np.ndarray
	observations: np.ndarray
	verification_state: np.ndarray
	verification_measurement: np.ndarray

	def __post_init__(self):

		self.init_fidelity_true = self.init_fidelity()
		self.init_fidelity_error = self.init_fidelity_error()
		self.init_fidelity_estimated = self.init_fidelity_estimated()
		self.init_fidelity_estimated_error = self.init_fidelity_estimated_error()
		self.number_of_iterations = last_non_nan_index(self.observations) + 1
		self.mean_number_of_iterations = self.mean_number_of_iterations()
		self.mean_number_of_iterations_error = self.mean_number_of_iterations_error()

	def init_fidelity(self):
		return np.mean(self.verification_state == 0)

	def init_fidelity_estimated(self):
		return np.mean(self.verification_measurement < 0.5)

	def init_fidelity_estimated_error(self):
		return np.std(self.verification_measurement < 0.5) / np.sqrt(self.verification_measurement.shape[0])

	def init_fidelity_error(self):
		return np.std(self.verification_state == 0) / np.sqrt(self.verification_state.shape[0])

	def mean_number_of_iterations(self):
		return np.mean(last_non_nan_index(self.observations)) + 1

	def mean_number_of_iterations_error(self):
		return np.std(last_non_nan_index(self.observations)) / np.sqrt(self.observations.shape[0])

	def plot_final_measurement(self):

		obseravtions = self.verification_measurement

		bins = np.linspace(obseravtions.min(), obseravtions.max(), 50)
		hist, bins = np.histogram(self.verification_measurement, bins = bins, density=False)

		number_of_observations = np.sum(hist)
		bin_width = np.mean(np.diff(bins))

		density = hist
		density_error = np.sqrt(hist)
		bin_centers = (bins[1:] + bins[:-1]) / 2

		plt.figure(f'Final measurement {datetime.now()}')
		plt.errorbar(bin_centers, density, yerr=2 * density_error, fmt='o')

		I_0 = gaussian(bin_centers, 0, self.model.std)
		I_1 = gaussian(bin_centers, 1, self.model.std)

		thermal_limit = I_0 * (1 - self.model.p_0_to_1) + I_1 * self.model.p_0_to_1
		thermal_eq = I_0 * self.model.steady_state[0] + I_1 * self.model.steady_state[1]

		plt.plot(bin_centers, thermal_limit * number_of_observations * bin_width, linestyle="--", c="k", label = "Thermal limit")
		plt.plot(bin_centers, thermal_eq * number_of_observations * bin_width, linestyle=":", c="k", label = "Thermal equilibrium")

		plt.xlabel("Final measurement")
		plt.ylabel("Counts")
		plt.yscale('log')
		plt.ylim([1, None])
		plt.legend()
		plt.show()

	def plot_number_of_iterations(self):
		plt.figure('Number of iterations')
		plt.hist(self.number_of_iterations, bins=range(1, self.number_of_iterations.max() + 1))
		plt.xlabel("Number of iterations")
		plt.ylabel("Frequency")
		plt.yscale('log')
		plt.show()

