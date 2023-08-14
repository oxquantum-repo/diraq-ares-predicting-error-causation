import numpy as np
import scipy.optimize
from scipy import stats

from errorcausation.active_reset import *
import scienceplots

np.random.seed(0)

plt.style.use(['science', 'no-latex', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(6, 4)

for i, p_init_0 in enumerate([0.508, 0.943]):

    qm_model = QubitInit(
        p_init_0=p_init_0,
        threshold=0.25,
        std=0.17,

        p_1_to_0=0.07,
        p_0_to_1=0.004,

        x_gate_fidelity=0.99,
    )

    hmm_model = QubitInit(
        p_init_0=p_init_0,
        threshold=0.5,
        std=0.17,

        p_1_to_0=0.07,
        p_0_to_1=0.004,

        x_gate_fidelity=0.99,
    )

    N = int(1e6)
    qm_results = qm_model.simulate_qm(N_repeat=N, iteration_max=10)
    hhm_results = hmm_model.simulate_hhm(N_repeat=N, iteration_max=10)
    hhm_gaussian_results = hmm_model.simulate_hhm_gaussian(N_repeat=N, iteration_max=10)

    print(
        f'qm fidelity estimated: {qm_results.init_fidelity_estimated:.4f} +/- {qm_results.init_fidelity_estimated_error:.4f}, '
        f'hhm fidelity: {hhm_results.init_fidelity_estimated:.4f} +/- {hhm_results.init_fidelity_estimated_error:.4f}'
        f'hhm fidelity gaussian: {hhm_gaussian_results.init_fidelity_estimated:.4f} +/- {hhm_gaussian_results.init_fidelity_estimated_error:.4f}'
    )


    print(
        f'QM iterations: {qm_results.mean_number_of_iterations:.4f} +/- {qm_results.mean_number_of_iterations_error:.4f}, '
        f'Catagorical HMM iterations: {hhm_results.mean_number_of_iterations:.4f} +/- {hhm_results.mean_number_of_iterations_error:.4f}'
        f'Gaussian HHM iterations: {hhm_gaussian_results.mean_number_of_iterations:.4f} +/- {hhm_gaussian_results.mean_number_of_iterations_error:.4f}'
    )

    qm_observation = qm_results.verification_measurement
    hmm_observation = hhm_results.verification_measurement
    hmm_gaussian_observation = hhm_gaussian_results.verification_measurement

    qm_bins = np.linspace(qm_observation.min(), qm_observation.max(), 50)
    qm_hist, qm_bins = np.histogram(qm_observation, bins = qm_bins, density=False)
    hhm_hist, _ = np.histogram(hmm_observation, bins = qm_bins, density=False)
    hhm_gaussian_hist, _ = np.histogram(hmm_gaussian_observation, bins = qm_bins, density=False)

    number_of_observations = np.sum(qm_hist)
    bin_width = np.mean(np.diff(qm_bins))

    qm_error = np.sqrt(qm_hist)
    hmm_error = np.sqrt(hhm_hist)
    hhm_gaussian_error = np.sqrt(hhm_gaussian_hist)

    bin_centers = (qm_bins[1:] + qm_bins[:-1]) / 2

    ax[i, 0].errorbar(bin_centers, qm_hist, yerr=2 * qm_error, fmt='.', label = f"QM: {100 * qm_results.init_fidelity_estimated:.2f}%")
    ax[i, 0].errorbar(bin_centers, hhm_hist, yerr=2 * hmm_error, fmt='.', label = f"Categorical HHM: {100 * hhm_results.init_fidelity_estimated:.2f}%")
    ax[i, 0].errorbar(bin_centers, hhm_gaussian_hist, yerr=2 * hhm_gaussian_error, fmt='.', label = f"Gaussian HHM: {100 * hhm_gaussian_results.init_fidelity_estimated:.2f}%")

    I_0 = gaussian(bin_centers, 0, qm_model.std)
    I_1 = gaussian(bin_centers, 1, qm_model.std)

    thermal_limit = I_0 * (1 - 0.5 * qm_model.p_0_to_1) + I_1 * 0.5 * qm_model.p_0_to_1
    initial = I_0 * qm_model.p_init_0 + I_1 * (1 - qm_model.p_init_0)

    y_initial = initial * number_of_observations * bin_width
    y_initial_error = np.sqrt(y_initial)

    ax[i, 0].errorbar(bin_centers, y_initial, yerr = 2 * y_initial_error, fmt='.', label = f"Initial: {100 * qm_model.p_init_0: .2f}%", c='red')
    ax[i, 0].plot(bin_centers, thermal_limit * number_of_observations * bin_width, linestyle="--", c="k",
                  label=f"Rethermalisation limit: {100 * (1 - 0.5 * qm_model.p_0_to_1): .2f}%")

    ax[i, 0].set_xlabel("$I$ validation (a.u)")
    ax[i, 0].set_ylabel("Counts")
    ax[i, 0].set_yscale('log')
    ax[i, 0].set_ylim([1, None])


    handles, labels = ax[i, 0].get_legend_handles_labels()
    ax[i, 0].legend(fontsize=7, handles=handles[::-1], labels=labels[::-1])

    qm_iterations = qm_results.number_of_iterations
    hmm_iterations = hhm_results.number_of_iterations
    hmm_gaussian_iterations = hhm_gaussian_results.number_of_iterations

    bins = np.arange(1, 100, 1)
    qm_hist, qm_bins = np.histogram(qm_iterations, bins = bins, density=True)
    hhm_hist, _ = np.histogram(hmm_iterations, bins = bins, density=True)
    hhm_gaussian_hist, _ = np.histogram(hmm_gaussian_iterations, bins = bins, density=True)

    bin_centers = (qm_bins[1:] + qm_bins[:-1]) / 2 - 0.5

    p_qm = 1 - np.cumsum(qm_hist)
    p_hhm = 1 - np.cumsum(hhm_hist)
    p_hhm_gaussian = 1 - np.cumsum(hhm_gaussian_hist)

    ax[i, 1].errorbar(bin_centers, p_qm, fmt='.', linestyle="--", label = f"QM: mean # iteration  {qm_results.mean_number_of_iterations :.2f}")
    ax[i, 1].errorbar(bin_centers, p_hhm, fmt='.', linestyle="--", label = f"Catagorical HHM: mean # iteration  {hhm_results.mean_number_of_iterations :.2f}")
    ax[i, 1].errorbar(bin_centers, p_hhm_gaussian, fmt='.', linestyle="--", label = f"Gaussian HHM: mean # iteration  {hhm_gaussian_results.mean_number_of_iterations :.2f}")

    ax[i, 1].set_yscale('log')
    ax[i, 1].set_ylim([1 / N, 1])
    ax[i, 1].set_xlim([0.9, 5.1])
    ax[i, 1].set_xticks([1, 2, 3, 4, 5])
    ax[i, 1].set_xticklabels([1, 2, 3, 4, "5+"])

    ax[i, 1].set_xlabel("# of iterations")
    ax[i, 1].set_ylabel("p(not terminated after \n # of iterations)")
    ax[i, 1].legend(fontsize=7)

for a,label in zip(ax.flatten(), 'abcdefghijklmnop'):
    a.text(-0.15, 1.1, f'({label})', transform=a.transAxes, fontweight='bold', va='top', ha='right')

fig.tight_layout()

plt.savefig(f"../figures/QM_vs_HMM_{datetime.now()}.pdf", bbox_inches='tight')
plt.show()
