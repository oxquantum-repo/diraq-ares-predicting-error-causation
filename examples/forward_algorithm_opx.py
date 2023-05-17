from time import sleep
import matplotlib.pyplot as plt
import numpy as np

from qm import QuantumMachinesManager

from errorcausation.opx.hmm_algorithms_opx import create_forward_program
from errorcausation.opx.hmm_algorithms_raw_python import forward
from errorcausation.Catagorical.categoricalmodel import CategoricalModel
# creating the model to simulate the data
model = CategoricalModel()
model.set_start_prob(0.5)
model.set_transition_prob(0.05, 0.02)
model.set_emission_prob(0.99, 0.99)

# simulating the data
measured_states, true_states = model.simulate_data(measurements=200, repeats=1)

# using the forward algorithm to compute the probability of the measured states given the data
p_calculated_on_computer = forward(measured_states, model.startprob_, model.transmat_, model.emissionprob_)

# creating the program to run on the OPX
forward_program = create_forward_program(measured_states, model.startprob_, model.transmat_, model.emissionprob_)

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
p_calculated_on_opx = np.stack([p0, p1], axis=1)

# comparing the results for the two methods
max_diff = np.abs(p_calculated_on_opx - p_calculated_on_computer).max()
print(f'Max difference between p_calculated_on_opx and p_calculated_on_computer is: {max_diff}')


# plotting the results
fig, ax = plt.subplots(3, 1, sharex=True)

ax[0].plot(1 - true_states.flatten(), label='True hidden spin state')
ax[0].plot(1 - measured_states.flatten(), label='Measured spin state')
ax[0].set_ylabel('Spin state')
ax[0].legend()

ax[1].plot(p_calculated_on_computer[:, 0], label='Calculated on Lab PC', c = 'r', alpha = 0.5)
ax[1].plot(p_calculated_on_opx[:, 0], label='Calculated OPX', c = 'k', alpha = 0.5)

ax[1].set_xlabel('Measurement number')
ax[1].set_ylabel('$p(even | measured spin states)$')
ax[1].legend()

diff = p_calculated_on_opx[:, 0] - p_calculated_on_computer[:, 0]
ax[2].plot(diff, label='Calculated on Lab PC', c = 'r', alpha = 0.5)
ax[2].set_ylabel('Difference between\n OPX and Lab PC')
ax[2].set_xlabel('Measurement number')

plt.show()
