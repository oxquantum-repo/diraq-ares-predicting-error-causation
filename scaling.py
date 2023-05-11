import matplotlib.pyplot as plt

from opx import *

# model = Model()
# probabilities = np.array([0.95, 0.02, 0.02, 0.99, 0.99])
# model.set_probabilities(*probabilities)
#
# O, spins = model.simulate_data(10, 1)

plt.figure('Forward')

ep = 1e-2

for tp in np.geomspace(1e-6, 1e-1, 6):
    O = np.zeros(15)

    S = np.array([0, 1])
    Pi = np.array([0.95, 0.5])
    Tm = np.array([
        [1 - tp, tp],
        [tp, 1 - tp]
    ])
    Em = np.array([
        [1 - ep, ep],
        [ep, 1 - ep]
    ])

    P = forward(O, S, Pi, Tm, Em)

    label = "$\log_{10} P($even$\\rightarrow$odd)$ =$" + "{}".format(np.log10(tp))
    plt.plot(100 * (1 - P[:, 0]), label =label)


plt.yscale('log')
plt.xlabel('Number of measurements')
plt.ylabel('Active initialisation infidelity (%)')
plt.legend()
plt.show()