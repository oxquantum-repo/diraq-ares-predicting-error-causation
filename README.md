# `errorcausation` - Diraq-Ares: Predicting Error Causation

A package for analysing and quantifying the SPAM + state flip errors in a quantum computer

## Quick start

### Installation

#### With `poetry`

1. Clone the repo

```bash
git clone  https://github.com/oxquantum-repo/diraq-ares-predicting-error-causation/
```

2. Set up your local python version with poetry

```bash
poetry env use 3.12
```

3. `cd` into the cloned repo directory `diraq-ares-predicting-error-causation/`

4. Install locally with poetry

```bash
poetry install
```

#### With `pip`

1. Clone the repo

```bash
git clone  https://github.com/oxquantum-repo/diraq-ares-predicting-error-causation/
```

2. In your terminal or anaconda prompt, create an environment and activate it, for example using anaconda

```bash
conda create --name errorcausation

conda activate errorcausation
```

3. `cd` into the cloned repo directory `diraq-ares-predicting-error-causation/`

4. Install the required packages using pip and the `errorcausation` pacakage

```bash
conda install pip
python3 -m pip install --upgrade build
python3 -m build
pip install -e .
```

The `-e` flags means that the package is in "developer/editable" mode, i.e. the changes that you make in the package will be reflected in your working environment

### Run Instructions

Below is an example of how one would run the `errorcausation` package:

```python
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from errorcausation.Categorical.categoricalmodel import CategoricalModel

np.random.seed(0)

file = Path('data/even_init.mat')
data = loadmat(file.resolve())
measured_states = 1 - data['measured_states'].squeeze()

# initialising a qm_model to fit to the data and setting the starting guess of parameters for the Baum-Welch algorithm
# to optimise
model_to_fit = CategoricalModel()
model_to_fit.set_start_prob(0.95)
model_to_fit.set_transition_prob(0.05, 0.05)
model_to_fit.set_emission_prob(0.95, 0.95)

# fitting the qm_model to the data, using the Baum-Welch algorithm. The uncertainty in the parameters is also computed
# using the Cramer-Rao lower bound.
model_to_fit.fit(measured_states, compute_uncertainty=True)

# printing the fitted qm_model, which should be close to the qm_model used to simulate the data
print(model_to_fit)

# using the fitted qm_model to predict the true qubit state from the measured state and plotting the results
predicted_states = model_to_fit.predict(measured_states, plot=True)

```

This was taken from [categorical_example_experimental.py](examples/diraq/categorical_example_experimental.py). More examples can be found in the [examples folder](examples/).

## Mission Statement

### Problem

![image](images/readout%20mock%20up%20example.png)

- Repeated PSB non-demolition readout (Two states: Even/Odd == (up, up; down, down)/(up, down; down, up)
- We try to initialise into the even state
- In Experiment Number 1 we measured: even, even, even, odd, odd

#### Questions to answer:

- What caused the change from even to odd in Experiment Number 1 for example?
- Was there a spin flip caused by the non-demolition measurement (back action)?
- Was there a readout error at iterations 4 and 5, i.e. the state was even, be we mistakenly read it as odd twice?
- Could the state have been initialised as odd instead of even, and we read it out incorrectly for the first 3 iterations as even?

### Objectives

- Given a sequence of readout data, predict the probability (with uncertainty) of:
  - A readout error
  - Spin flip
  - Initialisation error
- Therefore tell me what happened in this measurement based on these probabilities
- Infer the initialisation state (with uncertainty) based on readout data
- Predict the next state to be read out (with uncertainty) based on previous states

### Solution

- Treat the readout sequences as Markovian
- Use a Hidden Markov Model to extract the system probabilities
- Demonstrate success of Hidden Markov Model on simulated data

## Example results

### Simulated Data

![simulated data figure](images/simulated_data.png)

#### Model Performance

![model performance figure](images/model_performance.png)

### Real Data

![real data figure](images/real_data.png)

#### Model Performance

![model performance on real data figure](images/model_performance_on_real_data.png)

```bash
P_init: 0.990
P_spin_flip_even_to_odd: 0.016
P_spin_flip_odd_to_even: 0.048
P_readout: 0.935
```
