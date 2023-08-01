import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from datetime import datetime
from tqdm import tqdm

plt.style.use(['science', 'no-latex', 'grid', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

from errorcausation.active_reset import *

thresholds = np.linspace(-0.25, 1.5, 101)

fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(5, 3.5)

for p_init, label in zip([1., 0., 0.5], ['"ground"', '"excited"', 'superposition']):

	models = [
		QubitInit(
			p_init_0=p_init,
			threshold=threshold,
			std=0.17,

			p_1_to_0=0.05,
			p_0_to_1=0.004,

			x_gate_fidelity=0.99,
		) for threshold in thresholds
	]

	results = [model.simulate_qm(N_repeat=10000, iteration_max=1000) for model in tqdm(models)]

	mean_fidelity = np.stack([result.init_fidelity_estimated for result in results], axis=0)
	mean_number_of_iterations = np.stack([result.mean_number_of_iterations for result in results], axis=0)

	ax[0].plot(thresholds, 100 * (1 - mean_fidelity))
	ax[1].plot(thresholds, mean_number_of_iterations, label=label)

f_ground = np.stack([model.f_ground for model in models])
p_less_than_threshold = np.stack([model.p_less_than_threshold for model in models])

ax[0].axhline(y= 100 * 0.5 * models[0].p_0_to_1, color="black", linestyle="-", label="Rethermalisation limit")
ax[0].axhline(y= 100 * models[0].steady_state[1], color="black", linestyle="--", label="Steady state limit")
ax[0].plot(thresholds, 100 * (1 - f_ground), label="$p(0| I \leq I_{threshold})$", linestyle=":", c="k")

ax[0].set_xlabel("$I_{threshold}$  (a.u)")
ax[0].set_ylabel("Initialisation\ninfidelity (%)")
ax[0].set_yscale("log")
ax[0].set_ylim(0.05, 100)
ax[0].legend(fontsize=7)

ax[1].set_ylim(0, 10)
ax[1].plot(thresholds, 2 / p_less_than_threshold, label="$2 / p(I \leq I_{threshold}| 0 )$", linestyle='--', c="k")
ax[1].legend(fontsize=7)
ax[1].set_xlabel("$I_{threshold}$  (a.u)")
ax[1].set_ylabel("Number of\niterations")

x = np.linspace(-1, 2, 1000)

ax_0 = ax[0].twinx()
ax_0.fill_between(x, gaussian(x, 0, models[0].std), alpha=0.2)
ax_0.fill_between(x, gaussian(x, 1, models[0].std), alpha=0.2)
ax_0.set_ylim([0, 3])
ax_0.set_yticklabels([])
ax_0.set_yticks([])

ax_1 = ax[1].twinx()
ax_1.fill_between(x, gaussian(x, 0, models[0].std), alpha=0.2)
ax_1.fill_between(x, gaussian(x, 1, models[0].std), alpha=0.2)
ax_1.set_ylim([0, 3])
ax_1.set_yticklabels([])
ax_1.set_yticks([])

ax[0].set_xlim([-0.5, 1.5])

for a,label in zip(ax.flatten(), 'abcdefghijklmnop'):
    a.text(-0.15, 1.1, f'({label})', transform=a.transAxes, fontweight='bold', va='top', ha='right')

fig.tight_layout()
plt.savefig(f"../figures/active_reset_{datetime.now()}.pdf", bbox_inches='tight')