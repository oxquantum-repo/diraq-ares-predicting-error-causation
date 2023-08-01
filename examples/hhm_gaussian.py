from errorcausation.active_reset import *

qm_model = QubitInit(
    p_init_0=0.5,
    threshold=0.5,
    std=0.17,

    p_1_to_0=0.05,
    p_0_to_1=0.002,

    x_gate_fidelity=0.99,
)

qm = qm_model.simulate_qm(N_repeat=int(1e4), iteration_max=10)
categorical_hmm = qm_model.simulate_hhm(N_repeat=int(1e4), iteration_max=10)
gaussian_hhm = qm_model.simulate_hhm_gaussian(N_repeat=int(1e4), iteration_max=10)

print(
        f'qm fidelity estimated: {qm.init_fidelity_estimated:.4f} +/- {qm.init_fidelity_estimated_error:.4f},'
        f'categorical fidelity estimated: {categorical_hmm.init_fidelity_estimated:.4f} +/- {categorical_hmm.init_fidelity_estimated_error:.4f}, '
        f'gaussian fidelity: {gaussian_hhm.init_fidelity_estimated:.4f} +/- {gaussian_hhm.init_fidelity_estimated_error:.4f}'
    )

print(
        f'categorical fidelity estimated: {qm.mean_number_of_iterations:.4f} +/- {qm.mean_number_of_iterations_error:.4f}, '
        f'categorical fidelity estimated: {categorical_hmm.mean_number_of_iterations:.4f} +/- {categorical_hmm.mean_number_of_iterations_error:.4f}, '
        f'gaussian fidelity: {gaussian_hhm.mean_number_of_iterations:.4f} +/- {gaussian_hhm.mean_number_of_iterations_error:.4f}'
)