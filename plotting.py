import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = 1 - np.geomspace(0.001, 0.1, 10)
y = np.linspace(x.min(), x.max(), 10)

fig, ax = plt.subplots()
ax.plot(1 - x, y, 'o')
ax.set_xscale('log')
tick = ax.get_xticks()[::-1]
ax.set_xticklabels([f"{1 - tick:.2f}" for tick in tick])
