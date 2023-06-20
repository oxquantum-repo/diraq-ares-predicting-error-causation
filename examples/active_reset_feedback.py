import numpy as np

from errorcausation.Catagorical.categoricalmodel import CategoricalModel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

model = CategoricalModel()
model.set_start_prob(0.5)
model.set_transition_prob(0.001, 0.01)
model.set_emission_prob(0.99, 0.99)

gate_fidelity = 0.99

def decide_action(alpha, observation, transmat, emmisonprob):
    return alpha[0] < 0.01

def decide_to_terminate(alpha, observation, transmat, emmisonprob):
    return False

def forward(startprob, transmat, emmisonprob, N):
    """Forward algorithm for HMMs.
        O: observation sequence
        S: set of states
        Pi: initial state probabilities
        Tm: transition matrix
        Em: emission matrix
        """
    M = 2

    # the hidden states are the states of the HMM and are not accessible
    hidden_states = np.full(N, fill_value=-(N + 1), dtype=int)

    # the observations are the observations of the HMM and are accessible
    observations = np.full(N, fill_value=-(N + 1), dtype=int)

    # the actions are the actions of the HMM and are accessible
    actions = np.full(N, fill_value=-(N + 1), dtype=int)
    actions[0] = 0

    # computing the first hidden state and first observation
    hidden_states[0] = np.random.choice(2, p=startprob)
    observations[0] = np.random.choice(2, p=emmisonprob[hidden_states[0], :])

    # computing the first alpha, the probability of the hidden state given the observation
    alpha = np.full((N, M), fill_value=np.nan, dtype=float)
    alpha[0, :] = startprob * emmisonprob[:, observations[0]]
    alpha[0, :] /= np.sum(alpha[0, :])

    actions[1] = decide_action(alpha[0], observations[0], transmat, emmisonprob)

    for n in range(1, N):
        # computing the new hidden state
        p_hidden = transmat[actions[n], hidden_states[n - 1], :]
        hidden_states[n] = np.random.choice([0, 1], p=p_hidden)

        # computing the new observation based on the new hidden state
        p_observation = emmisonprob[hidden_states[n], :]
        observations[n] = np.random.choice([0, 1], p=p_observation)

        for m in range(M):
            alpha[n, m] = np.sum(alpha[n-1, :] * transmat[actions[n], :, m]) * emmisonprob[m, observations[n]]
        alpha[n, :] /= np.sum(alpha[n, :])

        if decide_to_terminate(alpha[n], observations[n], transmat, emmisonprob):
            for m in range(n, N):
                alpha[m, :] = alpha[n, :]
            break

        if n < N - 1:
            actions[n + 1] = decide_action(alpha[n], observations[n], transmat, emmisonprob)

    return hidden_states, observations, alpha, actions

def repeated_calls(startprob, transmat, emmisonprob, N, n_call):
    h = np.zeros((n_call, N), dtype=float)
    o = np.zeros((n_call, N), dtype=float)
    a = np.zeros((n_call, N, 2), dtype=float)
    actions = np.zeros((n_call, N), dtype=float)

    for i in range(n_call):
        hidden_states, observations, alpha, action = forward(startprob, transmat, emmisonprob, N)
        h[i, :] = hidden_states
        o[i, :] = observations
        a[i, :, :] = alpha
        actions[i, :] = action

    h[h < 0] = -1
    o[o < 0] = -1
    actions[actions < 0] = -1

    return h, o, a, actions


x_gate = np.array([
    [1 - gate_fidelity, gate_fidelity],
    [gate_fidelity, 1 - gate_fidelity]
])

transmat = np.stack([
    model.transmat_, x_gate
], axis=0)

hidden_states, observations, alpha, actions = repeated_calls(model.startprob_, transmat, model.emissionprob_, 20, 1000)

final_states = hidden_states[:, -1]
init_fidelity = (1 - final_states.mean()) * 100
init_fidelity_error = (1 - final_states).std() * 100 / np.sqrt(len(final_states))

final_alpha = alpha[:, -1, 0]
alpha_confidence_fidelity = (final_alpha).mean() * 100
alpha_confidence_fidelity_error = (final_alpha).std() * 100 / np.sqrt(len(final_alpha))


print(f'Initialization fidelity {init_fidelity :.6f} +/- {init_fidelity_error: .6f}% ')
print(f'Alpha confidence fidelity {alpha_confidence_fidelity :.6f} +/- {alpha_confidence_fidelity_error: .6f}% ')
fig, ax = plt.subplots(4, 1, sharex=True)


colour = (0.5, 0.1, 0.)
N = 100
ax[0].imshow(actions[0:N, :], aspect="auto", interpolation="antialiased", cmap=ListedColormap([colour,'black', 'white']))
ax[0].set_ylabel("Action")

ax[1].imshow(observations[0:N, :], aspect="auto", interpolation="antialiased", cmap=ListedColormap([colour,'black', 'white']))
ax[1].set_ylabel("Observation")

ax[2].imshow(alpha[0:N, :, 1], aspect="auto", interpolation="antialiased", cmap = 'hot')
ax[2].set_ylabel("Alpha")

ax[3].imshow(hidden_states[0:N, :], aspect="auto", interpolation="antialiased", cmap=ListedColormap([colour, 'black', 'white']))
ax[3].set_ylabel("Hidden state")


plt.show()