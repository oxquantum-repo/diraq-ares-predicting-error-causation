#%%
import glob, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib.colors import LogNorm
from decimal import Decimal

# data = loadmat("C:\\Users\\jonat\\OneDrive - UNSW\\Repeated_readout_project\\experimental data\\"+\
#                "Repeated_readout_1000_measurements_20_repeats_run_odd_init_18463.mat")
data = loadmat("C:\\Users\\jonat\\OneDrive - UNSW\\Repeated_readout_project\\experimental data\\"+\
                "Repeated_readout_1000_measurements_20_repeats_run_even_init_18433.mat")

#%%
repeat = data['repeats'].squeeze()
measurement = data['measurements'].squeeze()
measured_states = data['measured_states'].squeeze().T

bounds = [0.5, repeat+0.5, 0.5, measurement+0.5]

#%%
bin_cmap = matplotlib.colors.ListedColormap(['black', 'white'])

fig, ax = plt.subplots()
cax = ax.imshow(measured_states, aspect='auto', extent=bounds, interpolation='none',\
           origin='lower', cmap=bin_cmap)
cbar = fig.colorbar(cax, ticks=[0,1])
cbar.set_label('Outcome')
cbar.ax.set_yticklabels(['Odd','Even'])
plt.ylabel('# repeated readout')
plt.yticks(np.arange(1, measurement+1))
plt.xlabel('# experiment')
plt.show()
