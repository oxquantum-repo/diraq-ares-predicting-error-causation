import matplotlib.pyplot as plt

from src.opx import *
from qm.qua import *
from qualang_tools.loops import qua_arange
from qm.QuantumMachinesManager import QuantumMachinesManager
from src.opx import config, host
from time import sleep

from src import CategoricalModel

model = CategoricalModel()
probabilities = np.array([0.5, 0.002, 0.002, 0.9, 0.9])
model.set_probabilities(*probabilities)

O_, spins = model.simulate_data(200, 1)

S = np.array([0, 1])
Pi = model.startprob_
Tm = model.transmat_
Em = model.emissionprob_

true_alpha = forward(O_, S, Pi, Tm, Em)
true_alpha = true_alpha / np.sum(true_alpha, axis=1).reshape(-1, 1)


def ravel_index(n, m, shape_m):
    return n * shape_m + m


with program() as forward:
    N_ = len(O_)
    M_ = 2

    n = declare(int, value=0)
    m = declare(int, value=0)
    k = declare(int, value=0)

    sum = declare(fixed, value=0.)
    temp = declare(fixed, value=[0., 0.])
    recip = declare(fixed, value=0.)

    results_stream = declare_stream()

    N = declare(int, value=len(O_))
    M = declare(int, value=2)

    O = declare(int, value=O_.astype(int).tolist())
    Em = declare(fixed, value=Em.flatten().tolist())
    Tm = declare(fixed, value=Tm.flatten().tolist())

    alpha = declare(fixed, size=M_ * N_)
    assign(alpha[ravel_index(0, 0, M)], Pi[0] * Em[ravel_index(0, O[0], M)])
    assign(alpha[ravel_index(0, 1, M)], Pi[1] * Em[ravel_index(1, O[0], M)])

    with for_(*qua_arange(n, 1, N_, 1)):
        with for_(*qua_arange(m, 0, M_, 1)):
            assign(sum, 0.)
            with for_(*qua_arange(k, 0, M_, 1)):
                assign(sum, sum + alpha[ravel_index(n - 1, k, M)] * Tm[ravel_index(k, m, M)])
            assign(temp[m], sum * Em[ravel_index(m, O[n], M)])

        assign(recip, 1. / (temp[0] + temp[1]))
        assign(alpha[ravel_index(n, 0, M)], temp[0] * recip)
        assign(alpha[ravel_index(n, 1, M)], temp[1] * recip)

    with for_(m, 0, m < 2 * N, m + 1):
        save(alpha[m], results_stream)

    with stream_processing():
        results_stream.buffer(2 * N_).save("alpha")

qmm = QuantumMachinesManager(host=host)
qm = qmm.open_qm(config=config)

job = qm.execute(forward)

while job.result_handles.is_processing():
    sleep(0.01)

alpha = job.result_handles.get("alpha").fetch_all().reshape(-1, 2)
alpha = alpha / np.sum(alpha, axis=1).reshape(-1, 1)

print("the arrays are the same {}".format(np.isclose(alpha, true_alpha).all()))
print(np.abs(alpha - true_alpha).max())

plt.plot(1 - spins, label='True hidden spin state')
plt.plot(1 - O_, label='Measured spin state')

plt.plot(true_alpha[:, 0], label='Calculated on Lab PC', alpha=0.5)
plt.plot(alpha[:, 0], label='Calculated OPX')
plt.xlabel('Number of measurements')
plt.ylabel('P(Nth spin even|measurements)')
plt.legend()
plt.show()
