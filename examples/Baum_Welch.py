import matplotlib.pyplot as plt
import numpy as np

from errorcausation import CategoricalModel
from errorcausation.opx.categorical_hmm_algorithms_raw_python import baum_welch

# np.random.seed(0)

model = CategoricalModel()
model.set_start_prob(0.5)
model.set_transition_prob(0.1, 0.02)
model.set_emission_prob(0.99, 0.95)

start_prob = np.array([0.5, 0.5])
transition_prob = np.array([[0.99, 0.01], [0.01, 0.99]])
emission_prob = np.array([[0.99, 0.01], [0.01, 0.99]])

p_01 = []
p_10 = []

epsilon = 0.01

for i in range(100):
    p_01.append(transition_prob[0, 1])
    p_10.append(transition_prob[1, 0])

    measured_states, true_states = model.simulate_data(measurements=200, repeats=1, plot=False)
    startprob_after, transmat_after, emmisonprob_after = baum_welch(measured_states.squeeze(), start_prob, transition_prob, emission_prob, n_iter=1)

    start_prob = start_prob + (startprob_after - start_prob) * epsilon
    start_prob = start_prob / np.sum(start_prob, keepdims=True)

    transition_prob = transition_prob + (transmat_after - transition_prob) * epsilon
    transition_prob = transition_prob / np.sum(transition_prob, axis=1, keepdims=True)

    emission_prob = emission_prob + (emmisonprob_after - emission_prob) * epsilon
    emission_prob = emission_prob / np.sum(emission_prob, axis=1, keepdims=True)




plt.plot(p_01)
plt.plot(p_10)
plt.show()