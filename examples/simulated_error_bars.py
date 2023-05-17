import matplotlib.pyplot as plt

from errorcausation.Catagorical.categoricalmodel import CategoricalModel
from tqdm import tqdm

import numpy as np

model = CategoricalModel()
probabilities = np.array([0.999, 0.02, 0.02, 0.99, 0.99])
model.set_probs(*probabilities)

model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.99)
model_to_fit.set_transition_prob(0.02, 0.02)
model_to_fit.set_emission_prob(0.99, 0.99)

plt.axvline(model.get_start_prob(), color='k', linestyle='--')

for i in tqdm(range(10)):
    measured_states, true_states = model.simulate_data(20, repeats=1000)
    X = np.linspace(0.97, 1, 100)
    Y = np.zeros_like(X)
    for i, x in enumerate(X):
        model_to_fit.set_start_prob(x)
        Y[i] = (model_to_fit.score(measured_states))

    p_start_best, best_score = X[np.argmax(Y)], np.max(Y)
    model_to_fit.set_start_prob(p_start_best)
    model_to_fit.compute_uncertainty(measured_states)
    p_start, p_start_error = model_to_fit.get_start_prob_and_error()

    plt.plot(X, Y, alpha = 0.5)

    plt.errorbar(p_start, best_score, xerr=2 * p_start_error, fmt='x', color='k', capsize=5)
plt.xlabel('$p_{init}$')
plt.ylabel('$\log(L)$')
plt.show()

