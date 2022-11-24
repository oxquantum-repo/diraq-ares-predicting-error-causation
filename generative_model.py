# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import re
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from matplotlib.colors import LogNorm
import random
#%% Parameters
# Probability of initialising even state
P_init_even = 0.99
# Probability of a spin flipping from even to odd
P_spin_flip_even_to_odd = 0.02
# Probability of a spin flipping from odd to even
P_spin_flip_odd_to_even = 0.25
# Probability of reading out a spin incorrectly
P_charge_readout = 0.9995
#%% Initialise and readout operation
def initialise():
    initial_state = np.random.choice(np.arange(0, 2), p=[1-P_init_even, P_init_even])
    return initial_state

def readout(initial_state=None):
    if (np.random.choice(np.arange(0, 2), p=[1-P_charge_readout, P_charge_readout])):
        readout_signal = initial_state
    else:
        readout_signal = 1-initial_state
    return readout_signal
#%% Experiment run
Nshots = 100 # 200 # 500 #
Nread = 10
spin_state_array = np.zeros((Nread, Nshots))
readout_state_array = np.zeros((Nread, Nshots))
for i in range(0, Nshots):
    for j in range(0, Nread):
        if j<=0:
            # Before the 1st read, we initialise
            spin_state_array[j, i] = initialise()
        else:
            # After every read, we have some probability of spin flip
            if spin_state_array[j-1, i]==1:
                # if the spin has been even
                if (np.random.choice(np.arange(0, 2), p=[1-P_spin_flip_even_to_odd, P_spin_flip_even_to_odd])):
                    spin_state_array[j, i] = 0
                else:
                    spin_state_array[j, i] = spin_state_array[j-1, i]
            else:
                # if the spin has been odd
                if (np.random.choice(np.arange(0, 2), p=[1-P_spin_flip_odd_to_even, P_spin_flip_odd_to_even])):
                    spin_state_array[j, i] = 1
                else:
                    spin_state_array[j, i] = spin_state_array[j-1, i]
        readout_state_array[j, i] = readout(initial_state=spin_state_array[j, i])
#%% Plot
import matplotlib.colors
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=False)
bin_cmap = matplotlib.colors.ListedColormap(['black', 'white'])
bounds = [1, Nshots, 1, Nread]
fig, ax = plt.subplots()
h_plot = plt.imshow(readout_state_array, extent=bounds, aspect='auto', origin='lower', cmap=bin_cmap)
cbar = fig.colorbar(h_plot, ticks=[0,1])
cbar.set_label('Outcome')
cbar.ax.set_yticklabels(['Odd','Even'])
plt.xlabel('# shot')
plt.ylabel('# readout')
plt.show()