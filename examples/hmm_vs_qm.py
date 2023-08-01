from errorcausation.active_reset import *
import scienceplots


plt.style.use(['science', 'no-latex', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(6, 4)

for i, p_init_0 in enumerate([0.508, 0.943]):

    qm_model = QubitInit(
        p_init_0=p_init_0,
        threshold=0.5,
        std=0.17,

        p_1_to_0=0.05,
        p_0_to_1=0.003,

        x_gate_fidelity=0.99,
    )

    hmm_model = QubitInit(
        p_init_0=p_init_0,
        threshold=0.5,
        std=0.17,

        p_1_to_0=0.05,
        p_0_to_1=0.003,

        x_gate_fidelity=0.99,
    )

    qm_results = qm_model.simulate_qm(N_repeat=int(1e5), iteration_max=10)
    hhm_results = hmm_model.simulate_hhm(N_repeat=int(1e5), iteration_max=10)

    print(
        f'qm fidelity estimated: {qm_results.init_fidelity_estimated:.4f} +/- {qm_results.init_fidelity_estimated_error:.4f}, '
        f'hhm fidelity: {hhm_results.init_fidelity_estimated:.4f} +/- {hhm_results.init_fidelity_estimated_error:.4f}'
    )

    print(
        f'qm fidelity true: {qm_results.init_fidelity_true: .4f}, hhm fidelity: {hhm_results.init_fidelity_true}'
    )

    print(
        f'qm iterations: {qm_results.mean_number_of_iterations:.4f} +/- {qm_results.mean_number_of_iterations_error:.4f}, '
        f'hhm iterations: {hhm_results.mean_number_of_iterations:.4f} +/- {hhm_results.mean_number_of_iterations_error:.4f}'
    )


    qm_observation = qm_results.verification_measurement
    hmm_observation = hhm_results.verification_measurement

    qm_bins = np.linspace(qm_observation.min(), qm_observation.max(), 50)
    qm_hist, qm_bins = np.histogram(qm_observation, bins = qm_bins, density=False)
    hhm_hist, _ = np.histogram(hmm_observation, bins = qm_bins, density=False)


    number_of_observations = np.sum(qm_hist)
    bin_width = np.mean(np.diff(qm_bins))

    qm_error = np.sqrt(qm_hist)
    hmm_error = np.sqrt(hhm_hist)

    bin_centers = (qm_bins[1:] + qm_bins[:-1]) / 2


    ax[i, 0].errorbar(bin_centers, qm_hist, yerr=2 * qm_error, fmt='.', label = f"QM {qm_results.init_fidelity_estimated:.4f}%")
    ax[i, 0].errorbar(bin_centers, hhm_hist, yerr=2 * hmm_error, fmt='.', label = f"HHM {hhm_results.init_fidelity_estimated:.4f}%")

    I_0 = gaussian(bin_centers, 0, qm_model.std)
    I_1 = gaussian(bin_centers, 1, qm_model.std)

    thermal_limit = I_0 * (1 - 0.5 * qm_model.p_0_to_1) + I_1 * 0.5 * qm_model.p_0_to_1
    initial = I_0 * qm_model.p_init_0 + I_1 * (1 - qm_model.p_init_0)

    y_initial = initial * number_of_observations * bin_width
    y_initial_error = np.sqrt(y_initial)

    ax[i, 0].errorbar(bin_centers, y_initial, yerr = 2 * y_initial_error, fmt='.', label = f"Initial {qm_model.p_init_0: .4f}%", c='red')
    ax[i, 0].plot(bin_centers, thermal_limit * number_of_observations * bin_width, linestyle="--", c="k",
                  label="Thermal limit")

    ax[i, 0].set_xlabel("$I$ validation (a.u)")
    ax[i, 0].set_ylabel("Counts")
    ax[i, 0].set_yscale('log')
    ax[i, 0].set_ylim([1, None])
    ax[i, 0].legend(fontsize=7)

    plt.show()


    qm_iterations = qm_results.number_of_iterations
    hmm_iterations = hhm_results.number_of_iterations

    bins = np.arange(1, 8, 1)
    qm_hist, qm_bins = np.histogram(qm_iterations, bins = bins, density=False)
    hhm_hist, _ = np.histogram(hmm_iterations, bins = bins, density=False)

    error = np.sqrt(qm_hist + hhm_hist)

    bin_centers = (qm_bins[1:] + qm_bins[:-1]) / 2

    ax[i, 1].hist(qm_iterations, bins = bins, density=False, label = "QM", alpha = 0.5)
    ax[i, 1].hist(hmm_iterations, bins = bins, density=False, label = "HHM", alpha = 0.5)

    ax[i, 1].set_yscale('log')
    ax[i, 1].set_xlabel("Number of iterations")
    ax[i, 1].set_ylabel("Counts")

    ax_01 = ax[i, 1].twinx()

    ax_01.errorbar(bin_centers, hhm_hist - qm_hist, yerr = 2 * error, xerr = 0.5,  fmt='.', c="k", label = "Counts(HHM - QM)")
    ax[i, 1].axvline(x=hhm_results.mean_number_of_iterations, linestyle="--", label = "HHM mean")
    ax[i, 1].axvline(x=qm_results.mean_number_of_iterations, linestyle=":", label = "QM mean")

    ax[i, 1].set_xlabel("Number of iterations")
    ax_01.set_ylabel("Counts (HHM - QM)")
    ax[i, 1].legend(loc =  'upper right', fontsize = 7)
    ax_01.legend(loc = 'lower right', fontsize = 7)
    ax_01.axhline(y=0, linestyle="--", c="k", alpha = 0.5)

for a,label in zip(ax.flatten(), 'abcdefghijklmnop'):
    a.text(-0.15, 1.1, f'({label})', transform=a.transAxes, fontweight='bold', va='top', ha='right')

fig.tight_layout()

plt.savefig(f"../figures/QM_vs_HMM_{datetime.now()}.pdf", bbox_inches='tight')
plt.show()
