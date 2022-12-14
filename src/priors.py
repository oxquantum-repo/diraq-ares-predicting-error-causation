import numpy as np


def _calculate_readout_prior(measured_states):
	"""
	There are two ways a single pixel could be a different colour from its neighbours,
	either a spin flip followed by a spin flip back or a readout error, therefore
	
	p_different_from_neighbours = (1 - P_readout) + 2 * P_even_to_odd * P_odd_to_even
	
	If the backaction is weak, the probability of a spin flip followed by one back is negligable, therefore we can
	approximate
	
	p_different_from_neighbours approx P_readout_error
	
	:param measured_states: np.ndarray()
	:return: p_readout_prior: float
	"""
	# finding the single pixels which differ form their neighbours
	measurement_error = np.abs(np.diff(np.diff(measured_states.astype('float'), axis=1), axis=1)) // 2
	measurement_error[measurement_error < 1] = 0
	
	# calculating the probability a single pixel differs from its neighbours
	p_different_from_neighbours = measurement_error.mean()
	p_readout_prior = 1 - p_different_from_neighbours
	
	return p_readout_prior

def _calculate_spin_flip_even_to_odd_prior(measured_states, P_init_even_prior):
	"""
	the probability of measuring even consecutively for N measurements from initialisation is approximately
	(neglecting readout errors)
	P(no transitions) = P(init even) * (1 - P(even to odd))^N
	
	Therefore,
	P(even to odd) = 1 - (P(no transitions) / P(init even)) ^ (1 / N)

	:param measured_states:
	:return:
	"""
	p_no_transitions = np.all(measured_states == 0, axis=1).mean()
	p_spin_flip_even_to_odd_prior = 1 - (p_no_transitions / P_init_even_prior) ** (1 / measured_states.shape[1])
	return p_spin_flip_even_to_odd_prior

def _calculate_spin_flip_odd_to_even_prior(measured_states, p_spin_flip_even_to_odd_prior):
	"""
	If one considers a transition matrix of the form
	[[1 - P_eo, P_eo],
	[P_oe, 1 - P_oe]]
	
	The steadstate is [P_oe, P_eo] / (P_eo + P_oe). So the ratio of the occurrence of the even and odd state is
	P_oe / P_eo. So the probability a measurement is odd in the stead-state P_odd_ss = P_eo / (P_eo + P_oe) and
	P_oe = (1 / P_odd_ss - 1) * P_eo.
	
	We assume that the statistics of the last measurement of each sequence is in the stead state
	
	:param measured_states:
	:return:
	"""
	
	p_last_measurement_odd = measured_states[:, -1].mean()
	p_spin_flip_odd_to_even_prior = p_spin_flip_even_to_odd_prior * ((1 / p_last_measurement_odd) - 1)
	return p_spin_flip_odd_to_even_prior

def _calculate_init_even_prior(measured_states, P_readout_prior):
	
	"""
	The probability the first measurement is even is
	P(first measurement even) = P(init even) P(measure even | even) + P(init odd) P(measure even | odd)
	
	If the initialisation and readout fidelities are high then
	P(init even) P(measure even | even) >> P(init odd) P(measure even | odd)
	
	Meaning that the first term dominates and
	P(first measurement even) approx_eq P(init even) P(measure even | even)
	
	:param measured_states:
	:return:
	"""
	
	p_first_measurement_even = 1 - measured_states[:, 0].mean()
	p_init_even_prior = p_first_measurement_even / P_readout_prior
	if p_init_even_prior <= 1:
		return p_init_even_prior
	else:
		print(f"p_init_even_prior {p_init_even_prior} > 1 so using p_first_measurement_even {p_first_measurement_even} as the prior instead")
		return p_first_measurement_even
	
def calculate_priors(measured_states):
	"""
	A function to calculate the priors for each parameter
	:param measured_states:
	:return:
	"""
	p_readout_prior = _calculate_readout_prior(measured_states)
	p_init_even_prior = _calculate_init_even_prior(measured_states, p_readout_prior)
	p_spin_flip_even_to_odd_prior = _calculate_spin_flip_even_to_odd_prior(measured_states, p_init_even_prior)
	p_spin_flip_odd_to_even_prior = _calculate_spin_flip_odd_to_even_prior(measured_states, p_spin_flip_even_to_odd_prior)
	return p_readout_prior, p_spin_flip_even_to_odd_prior, p_spin_flip_odd_to_even_prior, p_readout_prior