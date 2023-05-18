import matplotlib.pyplot as plt

from errorcausation import CategoricalModel
from tqdm import tqdm

import numpy as np

model = CategoricalModel()
probabilities = np.array([0.99, 0.02, 0.02, 0.99, 0.99])
model.set_probs(*probabilities)

model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.99)
model_to_fit.set_transition_prob(0.02, 0.02)
model_to_fit.set_emission_prob(0.99, 0.99)

Ns = np.arange(100, 1100, 100, dtype=int)
measured_states, true_states = model.simulate_data(20, repeats=Ns.max())


functions = {
    'start_prob': ( lambda x: model_to_fit.set_start_prob(x), lambda: model_to_fit.get_start_prob_and_error()),
    'p_even_odd': ( lambda x: model_to_fit.set_transition_prob(x, probabilities[2]), lambda: (model_to_fit.get_transition_prob()[0], model_to_fit.get_transition_error()[0])),
    'p_odd_even': ( lambda x: model_to_fit.set_transition_prob(probabilities[1], x), lambda: (model_to_fit.get_transition_prob()[1], model_to_fit.get_transition_error()[1])),
    'f_even': ( lambda x: model_to_fit.set_emission_prob(x, probabilities[3]), lambda: (model_to_fit.get_emission_prob()[0], model_to_fit.get_emission_error()[0])),
    'f_odd': ( lambda x: model_to_fit.set_emission_prob(probabilities[4], x), lambda: (model_to_fit.get_emission_prob()[1], model_to_fit.get_emission_error()[1])),
}

setter_function, getter_function  = functions['start_prob']

for n in tqdm(Ns):
    X = np.linspace(0.96, 1., 10)
    Y = np.zeros_like(X)
    for i, x in enumerate(X):
        setter_function(x)
        Y[i] = (model_to_fit.score(measured_states[:n]))

    p_start_best, best_score = X[np.argmax(Y)], np.max(Y)

    setter_function(p_start_best)
    model_to_fit.compute_uncertainty(measured_states[:n])

    value, error = getter_function()
    plt.plot(X, Y, alpha = 0.5, label=f'{n} repeats')
    plt.errorbar(value, best_score, xerr=error, fmt='x', color='k', capsize=5)
    plt.legend()
plt.xlabel('$p_{init}$')
plt.ylabel('$\log(L)$')
plt.show()