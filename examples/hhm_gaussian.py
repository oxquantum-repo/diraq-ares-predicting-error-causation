from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from errorcausation.active_reset import *
from tqdm import tqdm
import scienceplots

plt.style.use(['science', 'no-latex', 'ieee', 'std-colors'])
plt.rcParams.update({'font.size': 10})

N = int(1e5)

qm_results = []
catagorical_hhm_results = []
gaussian_hhm_results = []

x = np.linspace(0.17, 0.3, 20)
for std in tqdm(x):
    hmm_model = QubitInit(
        p_init_0=0.5,
        threshold=0.5,
        std=std,

        p_1_to_0=0.07,
        p_0_to_1=0.004,

        x_gate_fidelity=0.99,
    )

    qm_model = deepcopy(hmm_model)
    qm_model.threshold = 0.25

    qm = qm_model.simulate_qm(N_repeat=N, iteration_max=10)
    categorical_hmm = hmm_model.simulate_hhm(N_repeat=N, iteration_max=10)
    gaussian_hhm = hmm_model.simulate_hhm_gaussian(N_repeat=N, iteration_max=10)

    qm_results.append(qm)
    catagorical_hhm_results.append(categorical_hmm)
    gaussian_hhm_results.append(gaussian_hhm)

# %% plotting

qm_fidelity = np.array([r.init_fidelity_true for r in qm_results])
qm_fidelity_error = np.array([r.init_fidelity_error for r in qm_results])
categorical_hhm_fidelity = np.array([r.init_fidelity_true for r in catagorical_hhm_results])
categorical_hhm_fidelity_error = np.array([r.init_fidelity_error for r in catagorical_hhm_results])
gaussian_hhm_fidelity = np.array([r.init_fidelity_true for r in gaussian_hhm_results])
gaussian_hhm_fidelity_error = np.array([r.init_fidelity_error for r in gaussian_hhm_results])

qm_number_of_iterations = np.array([r.mean_number_of_iterations for r in qm_results])
qm_number_of_iterations_error = np.array([r.mean_number_of_iterations_error for r in qm_results])

categorical_hhm_number_of_iterations = np.array([r.mean_number_of_iterations for r in catagorical_hhm_results])
categorical_hhm_number_of_iterations_error = np.array(
    [r.mean_number_of_iterations_error for r in catagorical_hhm_results])

gaussian_hhm_number_of_iterations = np.array([r.mean_number_of_iterations for r in gaussian_hhm_results])
gaussian_hhm_number_of_iterations_error = np.array([r.mean_number_of_iterations_error for r in gaussian_hhm_results])

ground_state_readout_fidelity = np.array([r.model.f_ground for r in qm_results])
best_fidelity = np.max(ground_state_readout_fidelity)

y_qm_fidelity = 100 * qm_fidelity * best_fidelity
y_qm_error = 100 * qm_fidelity_error * best_fidelity

y_categorical_hhm_fidelity = 100 * categorical_hhm_fidelity * best_fidelity
y_categorical_hhm_error = 100 * categorical_hhm_fidelity_error * best_fidelity

y_gaussian_hhm_fidelity = 100 * (gaussian_hhm_fidelity * best_fidelity)
y_gaussian_hhm_error = 100 * gaussian_hhm_fidelity_error * best_fidelity

fig, ax = plt.subplots(2, 2, gridspec_kw={'hspace': 0.1}, sharex='col', sharey='row')
fig.set_size_inches(6, 4)

ax[0, 0].errorbar(100 * ground_state_readout_fidelity, y_qm_fidelity, yerr=y_qm_error, label='QM', fmt='.',
                  linestyle='-')
ax[0, 0].errorbar(100 * ground_state_readout_fidelity, y_categorical_hhm_fidelity, yerr=y_categorical_hhm_error,
                  label='HHM', fmt='.', linestyle='-')
ax[0, 0].errorbar(100 * ground_state_readout_fidelity, y_gaussian_hhm_fidelity, yerr=y_gaussian_hhm_error,
                  label='HHM Gaussian', fmt='.', linestyle='-')
ax[0, 0].set_ylabel('Initialisation Fidelity (%)')

ax[1, 0].errorbar(100 * ground_state_readout_fidelity, qm_number_of_iterations, yerr=qm_number_of_iterations_error,
                  label='QM iterations', fmt='.', linestyle='-')
ax[1, 0].errorbar(100 * ground_state_readout_fidelity, categorical_hhm_number_of_iterations,
                  yerr=2 * categorical_hhm_number_of_iterations_error,
                  label='HHM iterations', fmt='.', linestyle='-')
errorbar = ax[1, 0].errorbar(100 * ground_state_readout_fidelity, gaussian_hhm_number_of_iterations,
                             yerr=2 * gaussian_hhm_number_of_iterations_error, label='HHM Gaussian iterations', fmt='.',
                             linestyle='-')
ax[1, 0].set_xlabel('Ground state discrimination fidelity\n$P(0|I \leq I_{threshold})$ (%)')
ax[1, 0].set_ylabel('Mean number of iterations')

qm_results = []
catagorical_hhm_results = []
gaussian_hhm_results = []

thresholds = 1 - np.geomspace(0.1, 1e-4, 20)
for termination_threshold in tqdm(thresholds):
    hmm_model = QubitInit(
        p_init_0=0.5,
        threshold=0.5,
        std=0.17,

        p_1_to_0=0.07,
        p_0_to_1=0.004,

        x_gate_fidelity=0.99,
    )
    hmm_model.termination_threshold = termination_threshold

    qm_model = deepcopy(hmm_model)
    qm_model.threshold = 0.25

    qm = qm_model.simulate_qm(N_repeat=N, iteration_max=100)
    categorical_hmm = hmm_model.simulate_hhm(N_repeat=N, iteration_max=100)
    gaussian_hhm = hmm_model.simulate_hhm_gaussian(N_repeat=N, iteration_max=100)

    qm_results.append(qm)
    catagorical_hhm_results.append(categorical_hmm)
    gaussian_hhm_results.append(gaussian_hhm)

# %% plotting

qm_fidelity = np.array([r.init_fidelity_true for r in qm_results])
qm_fidelity_error = np.array([r.init_fidelity_error for r in qm_results])
categorical_hhm_fidelity = np.array([r.init_fidelity_true for r in catagorical_hhm_results])
categorical_hhm_fidelity_error = np.array([r.init_fidelity_error for r in catagorical_hhm_results])
gaussian_hhm_fidelity = np.array([r.init_fidelity_true for r in gaussian_hhm_results])
gaussian_hhm_fidelity_error = np.array([r.init_fidelity_error for r in gaussian_hhm_results])

qm_number_of_iterations = np.array([r.mean_number_of_iterations for r in qm_results])
qm_number_of_iterations_error = np.array([r.mean_number_of_iterations_error for r in qm_results])

categorical_hhm_number_of_iterations = np.array([r.mean_number_of_iterations for r in catagorical_hhm_results])
categorical_hhm_number_of_iterations_error = np.array(
    [r.mean_number_of_iterations_error for r in catagorical_hhm_results])

gaussian_hhm_number_of_iterations = np.array([r.mean_number_of_iterations for r in gaussian_hhm_results])
gaussian_hhm_number_of_iterations_error = np.array([r.mean_number_of_iterations_error for r in gaussian_hhm_results])

ground_state_readout_fidelity = np.array([r.model.f_ground for r in qm_results])
best_fidelity = np.max(ground_state_readout_fidelity)

y_qm_fidelity = 100 * qm_fidelity * best_fidelity
y_qm_error = 100 * qm_fidelity_error * best_fidelity

y_categorical_hhm_fidelity = 100 * categorical_hhm_fidelity * best_fidelity
y_categorical_hhm_error = 100 * categorical_hhm_fidelity_error * best_fidelity

y_gaussian_hhm_fidelity = 100 * (gaussian_hhm_fidelity * best_fidelity)
y_gaussian_hhm_error = 100 * gaussian_hhm_fidelity_error * best_fidelity

thresholds = (1 - thresholds)[::-1]

ax[0, 1].errorbar(100 * thresholds, y_qm_fidelity, yerr=y_qm_error, label='QM', fmt='.',
                  linestyle='-')
ax[0, 1].errorbar(100 * thresholds, y_categorical_hhm_fidelity, yerr=y_categorical_hhm_error,
                  label='Categorical HHM', fmt='.', linestyle='-')
ax[0, 1].errorbar(100 * thresholds, y_gaussian_hhm_fidelity, yerr=y_gaussian_hhm_error,
                  label='Gaussian HHM', fmt='.', linestyle='-')

ax[1, 1].errorbar(100 * thresholds, qm_number_of_iterations, yerr=qm_number_of_iterations_error,
                  label='QM', fmt='.', linestyle='-')
ax[1, 1].errorbar(100 * thresholds, categorical_hhm_number_of_iterations,
                  yerr=categorical_hhm_number_of_iterations_error,
                  label='Categorical HHM', fmt='.', linestyle='-')
errorbar = ax[1, 1].errorbar(100 * thresholds, gaussian_hhm_number_of_iterations,
                             yerr=gaussian_hhm_number_of_iterations_error, label='Gaussian HHM', fmt='.',
                             linestyle='-')
ax[1, 1].set_xlabel('Termination threshold (%)')
ax[1, 1].legend()

ax[0, 1].set_xscale('log')
ax[1, 1].set_xscale('log')

tick = ax[0, 1].get_xticks()[::-1]
ax[0, 1].set_xticklabels([f"{100 - tick:.2f}" for tick in tick])
ax[1, 1].set_xticklabels([f"{100 - tick:.2f}" for tick in tick])

fig.tight_layout()
plt.savefig(f'../figures/hhm_gaussian_{datetime.now()}.pdf', bbox_inches='tight')
