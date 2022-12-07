
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.io import loadmat
from pathlib import Path
from hmmlearn import hmm
from tqdm import tqdm

file = Path("./data/200_repeats_20_measurements.mat")
data = loadmat(file)

#%%
repeat = data.get('repeat').squeeze()
measurement = data.get('measurement').squeeze()
measured_states = 1 - data.get('measured_states').squeeze().T

measurements = measured_states.shape[1]
repeats = measured_states.shape[0]

#%%

# plotting the generated data
plt.imshow(measured_states.T.squeeze(),
		   cmap='Greys', aspect='auto',
		   origin='lower', interpolation='none'
		   )
plt.xlabel('# repeat')
plt.ylabel('# measurement')
plt.draw()

# %%  coming up with heuristic priors for the parameters

# the probability the first measurement is even is
# P(first measurement even) = P(init even) P(measure even | even) + P(init odd) P(measure even | odd)
# if the initialisation and readout fidelities are any good then
# P(init even) P(measure even | even) >> P(init odd) P(measure even | odd)
# P(first measurement even) approx_eq P(init even) P(measure even | even)
# very heuristic we can assume P(init even) = P(measure even | even) therefore
# P(first measurement even) = P(init even)^2 = P(measure even | even)^2

P_first_measurement_even = 1 - measured_states[:, 0].mean()
P_init_even_prior = np.sqrt(P_first_measurement_even)
P_readout_pior = np.sqrt(P_first_measurement_even)

# the probability of measuring even consecutively for N measurements from initialisation is approximately
# (neglecting readout errors):
# P(no transitions) = P(init even) * (1 - P(even to odd))^N
# P(even to odd) = 1 - (P(no transitions) / P(init even)) ^ (1 / N)
P_no_transitions = np.all(measured_states == 0, axis=1).mean()
P_spin_flip_even_to_odd_prior = 1 - (P_no_transitions / P_init_even_prior) ** (1 / measurements)

# if one considers a transition matrix of the form
# [[1 - P_eo, P_eo],
#  [P_oe, 1 - P_oe]]
# the stead state is [P_oe, P_eo] / (P_eo + P_oe). So the ratio of the occurrence of the even and odd state is
# P_oe / P_eo. So the probability a measurement is odd in the stead-state P_odd_ss = P_eo / (P_eo + P_oe) and
# P_oe = (1 / P_odd_ss - 1) * P_eo.
# We assume that the statistics of the last measurement of each sequence is in the stead state

P_last_measurement_odd = measured_states[:, -1].mean()
P_spin_flip_odd_to_even_prior = P_spin_flip_even_to_odd_prior * ((1 / P_last_measurement_odd) - 1)

# %%  fitting hidden markov models to the data


# in principle there could be a lot of data, which would be computationally expensive to fit to. So we randomly select
# a subset of the data to fit to. The variable "sequences_in_subset" sets how many initialisation, measurement, ...
# sequences are included in the subset. For each of these subsets we fit a hidden markov model with a random
# initialisation informed by our priors.

sequences_in_subset = 200
number_of_models_to_fit = 50

# an array to inform the fit about the shape of the data, aka how many measurements are in each sequence
subset_shapes = np.full(shape=sequences_in_subset, fill_value=measurements)
full_shapes = np.full(shape=repeats, fill_value=measurements)

models = []
for _ in tqdm(range(number_of_models_to_fit)):
    # creating the hidden markov model
    model = hmm.CategoricalHMM(n_components=2, init_params='')
    model.n_features = 2
    
    # setting the initial parameters of our hmm model somewhat randomly according to our priors on the parameters
    model.startprob_ = np.random.dirichlet([P_init_even_prior, 1 - P_init_even_prior])
    model.transmat_ = np.array([np.random.dirichlet([1 - P_spin_flip_even_to_odd_prior, P_spin_flip_even_to_odd_prior]),
                                np.random.dirichlet(
                                    [P_spin_flip_odd_to_even_prior, 1 - P_spin_flip_odd_to_even_prior])])
    
    model.emissionprob_ = np.array([np.random.dirichlet([P_readout_pior, 1 - P_readout_pior]),
                                    np.random.dirichlet([1 - P_readout_pior, P_readout_pior])])
    
    # creating the random subset of the data
    random_subset_indices = np.random.choice(repeats, sequences_in_subset)
    random_subset = measured_states[random_subset_indices, ...].reshape(-1, 1)
    
    # fitting the model to the subset of the data
    model.fit(random_subset, subset_shapes)
    
    # storing the score of the fitted model, evaluated over the whole dataset
    model.score = model.score(measured_states.reshape(-1, 1), full_shapes)
    models.append(model)

# %% plotting the distribution of the parameters for the maximum likelihood fit. The variation arises from two sources
# 1. the randomness of the initial conditions resulting in the optimiser converging to different locations
# 2. the randomness of the subset of data

# extracting the fit parameters and associated score for each of the fitted models
start_prob = np.array([model.startprob_ for model in models])
transmat = np.array([model.transmat_ for model in models])
emisprob = np.array([model.emissionprob_ for model in models])
scores = np.array([model.score for model in models])

# finding the model which fits the data best
best_model = models[np.argmax(scores)]
# fitting the model to the complete dataset just incase it changes the minimum slightly
best_model.fit(measured_states.reshape(-1, 1), full_shapes)

# taking the parameters out of matrix form
P_init_even_estimates = start_prob[:, 0]
P_spin_flip_even_to_odd_estimates = transmat[:, 0, 1]
P_spin_flip_odd_to_even_estimates = transmat[:, 1, 0]
P_readout_estimate = (emisprob[:, 0, 0] + emisprob[:, 1, 1]) / 2

fig, ax = plt.subplots(nrows=2, ncols=2)

P_init_even_inferred = best_model.startprob_[0]
P_spin_flip_even_to_odd_inferred = best_model.transmat_[0, 1]
P_spin_flip_odd_to_even_inferred= best_model.transmat_[1, 0]
P_readout_inferred = (best_model.emissionprob_[0, 0] + best_model.emissionprob_[1, 1]) / 2

# actually plotting
ax[0, 0].hist(P_init_even_estimates, bins=20, color='b')
ax[0, 0].axvline(P_init_even_prior, c='k', linestyle=':', label='prior')
ax[0, 0].axvline(P_init_even_inferred, c='k', linestyle='-.', label='inferred')
ax[0, 0].set_xlabel("P(init_even) \n estimate")
ax[0, 0].legend(loc=0)
ax[0, 0].set_ylabel('Counts')

ax[0, 1].hist(P_spin_flip_even_to_odd_estimates, bins=20, color='g')
ax[0, 1].axvline(P_spin_flip_even_to_odd_prior, c='k', linestyle=':', label='prior')
ax[0, 1].axvline(P_spin_flip_even_to_odd_inferred, c='k', linestyle='-.', label='inferred')
ax[0, 1].set_xlabel("P(spin_flip \n even_to_odd) \n estimates")
ax[0, 1].legend(loc=0)
ax[0, 1].set_ylabel('Counts')

ax[1, 0].hist(P_spin_flip_odd_to_even_estimates, bins=20, color='r')
ax[1, 0].axvline(P_spin_flip_odd_to_even_prior, c='k', linestyle=':', label='prior')
ax[1, 0].axvline(P_spin_flip_even_to_odd_inferred, c='k', linestyle='-.', label='inferred')
ax[1, 0].set_xlabel("P(spin_flip \n odd_to_even) \n estimates")
ax[1, 0].legend(loc=0)
ax[1, 0].set_ylabel('Counts')

ax[1, 1].hist(P_readout_estimate, bins=50, color='g')
ax[1, 1].axvline(P_readout_pior, c='k', linestyle=':', label='prior')
ax[1, 1].axvline(P_readout_inferred,
	c='k', linestyle='-.', label='inferred')
ax[1, 1].set_xlabel("P(readout correct) \n estimates")
ax[1, 1].legend(loc=0)
ax[1, 1].set_ylabel('Counts')

fig.tight_layout()
plt.show()


print(f"P_init: {P_init_even_inferred :.3f}")
print(f"P_spin_flip_even_to_odd: {P_spin_flip_even_to_odd_inferred :.3f}")
print(f"P_spin_flip_odd_to_even: {P_spin_flip_odd_to_even_inferred :.3f}")
print(f"P_readout: {P_readout_inferred :.3f}")