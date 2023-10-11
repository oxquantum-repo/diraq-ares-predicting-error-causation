import matplotlib.pyplot as plt

from errorcausation import CategoricalModel
from tqdm import tqdm

import scienceplots

import numpy as np

np.random.seed(0)

plt.style.use(["science", "no-latex", "grid", "ieee", "std-colors"])
plt.rcParams.update({"font.size": 10})

probs = {
    "start_prob": 0.99,
    "p_even_odd": 0.01,
    "p_odd_even": 0.02,
    "f_even": 0.995,
    "f_odd": 0.99,
}

model = CategoricalModel()
probabilities = np.array([*probs.values()])
model.set_probs(*probabilities)

model_to_fit = CategoricalModel()
model_to_fit.set_probs(*probabilities)

Ns = np.arange(100, 1100, 100, dtype=int)
measured_states, true_states = model.simulate_data(20, repeats=Ns.max())

functions = {
    "start_prob": (
        lambda x: model_to_fit.set_start_prob(x),
        lambda: model_to_fit.get_start_prob_and_error(),
    ),
    "p_even_odd": (
        lambda x: model_to_fit.set_transition_prob(x, probabilities[2]),
        lambda: (
            model_to_fit.get_transition_prob()[0],
            model_to_fit.get_transition_error()[0],
        ),
    ),
    "p_odd_even": (
        lambda x: model_to_fit.set_transition_prob(probabilities[1], x),
        lambda: (
            model_to_fit.get_transition_prob()[1],
            model_to_fit.get_transition_error()[1],
        ),
    ),
    "f_even": (
        lambda x: model_to_fit.set_emission_prob(x, probabilities[3]),
        lambda: (
            model_to_fit.get_emission_prob()[0],
            model_to_fit.get_emission_error()[0],
        ),
    ),
    "f_odd": (
        lambda x: model_to_fit.set_emission_prob(probabilities[4], x),
        lambda: (
            model_to_fit.get_emission_prob()[1],
            model_to_fit.get_emission_error()[1],
        ),
    ),
}

ranges = {
    "start_prob": (0.97, 1.0),
    "p_even_odd": (0.00, 0.03),
    "p_odd_even": (0.00, 0.03),
    "f_even": (0.97, 1.0),
    "f_odd": (0.97, 1.0),
}

fig, ax = plt.subplots(1, 5, sharey=True)
fig.set_size_inches(6, 4)

for a, name, label in zip(
    ax,
    ["start_prob", "p_even_odd", "p_odd_even", "f_even", "f_odd"],
    [
        "$P_{\mathrm{init, even}}$",
        "$P_{\mathrm{even \\rightarrow odd}}$",
        "$P_{\mathrm{odd \\rightarrow even}}$",
        "$P_{\mathrm{read, even}}$",
        "$P_{\mathrm{read, odd}}$",
    ],
):
    setter_function, getter_function = functions[name]

    # Create a list of grayscale colors
    _colors = [str(i / 12) for i in range(1, 11)]

    for _i, n in enumerate(tqdm(Ns)):
        X = np.linspace(*ranges[name], 100)
        Y = np.zeros_like(X)
        for i, x in enumerate(X):
            setter_function(x)
            Y[i] = model_to_fit.score(measured_states[:n])

        p_start_best, best_score = X[np.argmax(Y)], np.max(Y)

        setter_function(p_start_best)
        model_to_fit.compute_uncertainty(measured_states[:n])

        value, error = getter_function()
        a.plot(X, Y, color=_colors[_i], label=f"{n} repeats")
        a.axvline(probs[name], color="k", linestyle="--")
        a.set_xlim(probs[name] - 0.01, probs[name] + 0.01)

        xerror = [[min(value, error)], [min(1.0 - value, error)]]
        a.errorbar(
            value, best_score, xerr=xerror, fmt=".", color="k", capsize=5, markersize=5
        )

    a.set_xlabel(label)
    a.set_ylim(-2000, 0)

ax[0].set_ylabel("Log-likelihood")

for a, label in zip(ax, "abcdefghijklmnop"):
    a.text(
        0.05,
        1.1,
        f"({label})",
        transform=a.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.tight_layout()
file_path = "."
plt.savefig(f"{file_path}/log_likelihood_error_bars.pdf", bbox_inches="tight")

plt.show()
