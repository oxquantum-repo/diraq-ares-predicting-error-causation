import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter_ns, sleep

from qm import QuantumMachinesManager

from errorcausation import GaussianModel
from errorcausation.opx.gaussian_hmm_algorithms_raw_python import forward
from errorcausation.opx.gaussian_hmm_algorithms_opx import create_forward_program

model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.8, 0.2])

model.transmat_ = np.array([[0.97, 0.03],
                            [0.1, 0.9]])

model.means_ = np.array([[0.0, 0.0], [1, 1]]) / np.sqrt(2)
model.covars_ = np.array([0.75, 0.75]) ** 2

X, Z = model.simulate_data(200, repeats = 1, plot=False)

forward_program = create_forward_program(
    X.squeeze(),
    np.array([0.5, 0.5]),
    model.transmat_,
    model.means_,
    model.covars_
)

t0 = perf_counter_ns()
# using the forward algorithm to compute the probability of the measured states given the data

alpha = forward(X,
    np.array([0.5, 0.5]),
    model.transmat_,
    model.means_,
    model.covars_
)

t1 = perf_counter_ns()

# creating a blank config file
host = "129.67.84.123"
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {},
            "digital_outputs": {},
            "analog_inputs": {},
        },
    },
    "elements": {},
    "pulses": {},
    "waveforms": {},
    "digital_waveforms": {},
    "integration_weights": {},
    "mixers": {},
}

# running the program on the OPX
qmm = QuantumMachinesManager(host=host)
qm = qmm.open_qm(config=config)
job = qm.execute(forward_program)

while job.result_handles.is_processing():
    sleep(0.01)

# fetching the results from the OPX
p0 = job.result_handles.get("p0").fetch_all()
p1 = job.result_handles.get("p1").fetch_all()
alpha_opx = np.stack([p0, p1], axis=1)

p0_timestamps = job.result_handles.get("p0_timestamps").fetch_all()
p1_timestamps = job.result_handles.get("p1_timestamps").fetch_all()

p0_compute_latency = np.diff(p0_timestamps).max()
p1_compute_latency = np.diff(p1_timestamps).max()
print(f'Total PC compute time: {(t1 - t0) / 1000} us')
print(f'Total OPX compute time: {max(p0_timestamps.max(), p1_timestamps.max()) / 1000} us')
print(f'p0, p1 compute latency: {p0_compute_latency, p1_compute_latency} ns')

fig, ax = plt.subplots(3, 1)
ax[0].plot(X[0, :, 0], label = "I")
ax[0].plot(X[0, :, 1], label = "Q")

ax[1].plot(alpha[:, 1], label = "p(excited)")
ax[1].plot(alpha_opx[:, 1], label ="p(excited) on OPX")
ax[1].plot(Z[0, :], label = "true state")
ax[1].legend()

ax[2].plot(alpha[:, 1] - alpha_opx[:, 1], label = "difference")
fig.tight_layout()

plt.show()