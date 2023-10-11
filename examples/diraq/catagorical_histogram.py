from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import scienceplots

from errorcausation.Catagorical.categoricalmodel import CategoricalModel

plt.style.use(["science", "no-latex", "grid", "ieee", "std-colors"])
plt.rcParams.update({"font.size": 10})
np.random.seed(0)

# initialising a qm_model to simulate data which we will fit another
model = CategoricalModel()
model.set_start_prob(0.99)
model.set_transition_prob(0.01, 0.02)
model.set_emission_prob(0.995, 0.99)

fit_models = []

for _ in tqdm(range(100)):
    # using the qm_model to simulate data and plotting it.
    # the number of measurements is the number of measurements to perform before the qubit is reset
    measured_states, true_states = model.simulate_data(measurements=20, repeats=1000)

    # initialising a qm_model to fit to the data and setting the starting guess of parameters for the Baum-Welch algorithm
    # to optimise
    model_to_fit = CategoricalModel()
    model_to_fit.set_start_prob(0.90)
    model_to_fit.set_transition_prob(0.1, 0.1)
    model_to_fit.set_emission_prob(0.90, 0.90)

    # fitting the qm_model to the data, using the Baum-Welch algorithm. The uncertainty in the parameters is also computed
    # using the Cramer-Rao lower bound.
    model_to_fit.fit(measured_states, compute_uncertainty=False)

    fit_models.append(model_to_fit)

model.compute_uncertainty(measured_states)

f_init = np.array([model.get_start_prob() for model in fit_models])
f_emission = np.array([model.get_emission_prob() for model in fit_models])
f_transition = np.array([model.get_transition_prob() for model in fit_models])

fig, ax = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(6, 2.5)


def height(std):
    return np.exp(-0.5) / (std * np.sqrt(2 * np.pi))


density = True
N_bin = 40

ax[0].hist(f_init, bins=N_bin, density=density, label ="$P_{init, even}$", alpha = 0.5, color="blue")
error = model.get_start_error()
ax[0].errorbar(
    model.get_start_prob(), height(error), xerr=error, color="black", capsize=3
)
ax[0].axvline(model.get_start_prob(), color="black", linestyle="--")
ax[0].legend()
ax[0].set_ylabel("Probability density")
ax[0].set_xlabel("Probability")

trans_bins = np.linspace(f_transition.min(), f_transition.max(), N_bin)
ax[1].hist(f_transition[:, 0], bins=trans_bins, density=density, alpha = 0.5, label ="$P_{even \\rightarrow odd}$", color="red")
ax[1].hist(f_transition[:, 1], bins=trans_bins, density=density, alpha = 0.5, label ="$P_{odd \\rightarrow even}$", color="green")
for i in range(2):
    error = model.get_transition_error()[i]
    ax[1].axvline(model.get_transition_prob()[i], color="black", linestyle="--")
    ax[1].errorbar(
        model.get_transition_prob()[i],
        height(error),
        xerr=error,
        color="black",
        capsize=3,
    )
ax[1].legend()
ax[1].set_xlabel("Probability")

emission_bins = np.linspace(f_emission.min(), f_emission.max(), N_bin)
ax[2].hist(f_emission[:, 0], bins=emission_bins, density=density, alpha = 0.5, label ="$P_{read, even}$", color ='purple')
ax[2].hist(f_emission[:, 1], bins=emission_bins, density=density, alpha = 0.5, label ="$P_{read, odd}$", color ='orange')
ax[2].axvline(model.get_emission_prob()[0], color="black", linestyle="--")
ax[2].axvline(model.get_emission_prob()[1], color="black", linestyle="--")
ax[2].set_xlabel("Probability")

for i in range(2):
    error = model.get_emission_error()[i]
    ax[2].axvline(model.get_emission_prob()[i], color="black", linestyle="--")
    ax[2].errorbar(
        model.get_emission_prob()[i],
        height(error),
        xerr=model.get_emission_error()[i],
        color="black",
        capsize=3,
    )

ax[2].legend()

for i in range(3):
    locs = ax[i].get_yticks()
    ax[i].set_yticks(locs, np.round(locs / len(f_init), 3))

for a, label in zip(ax, "abcdefghijklmnop"):
    a.text(
        -0.1,
        1.1,
        f"({label})",
        transform=a.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.tight_layout()

file_path = "."

plt.savefig(f"{file_path}/catagorical_histogram.pdf", bbox_inches="tight")
plt.show()
