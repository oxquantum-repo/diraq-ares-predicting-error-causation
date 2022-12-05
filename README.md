# Diraq-Ares: Predicting Error Causation

Main script: https://github.com/oxquantum-repo/diraq-ares-predicting-error-causation/blob/main/main.py

## Quick start

### Installation

1. Clone the repo 

2. In your terminal or anaconda prompt, create an environment and activate it, for example using anaconda

```bash
conda create --name error-causation

conda activate error-causation
```

3. `cd` into the cloned repo directory `diraq-ares-predicting-error-causation/`

4. Install the required packages using pip

```bash
conda install pip

pip install -r requirements.txt
```

### Run

To run the script run and test the model on simulated data `cd` into the cloned repo directory `diraq-ares-predicting-error-causation/` and run:

```python
python main.py
```

In your terminal or ipython prompt/interactive development environment of choice. There will plots generated. 

## `main.py` Walkthrough

- Creating the generative model

```python
P_init_even = 0.99  # the probability of initialising in the even state
P_spin_flip_even_to_odd = 0.02  # the probability of back-action flipping the state from even to odd
P_spin_flip_odd_to_even = 0.25  # the probability of back-action flipping the state from odd to even
P_readout = 0.999  # the probability of correctly reading out the state

gen_model = hmm.CategoricalHMM(n_components=2)  # a hidden markov model (hmm) with 2 components
```

- the probability of initialising in the two states [even, odd]
gen_model.startprob_ = np.array([P_init_even, 1 - P_init_even])

- the transition matrix [[even-even, even-odd], [odd-even, odd-odd]]
gen_model.transmat_ = np.array([
    [1 - P_spin_flip_even_to_odd, P_spin_flip_even_to_odd],
    [P_spin_flip_odd_to_even, 1 - P_spin_flip_odd_to_even]
])

- the emission matrix encoding the readout fidelities
- [[readout even when even, readout even when odd],
- [readout odd when even, readout odd when odd]]
```python 
gen_model.emissionprob_ = np.array([
    [P_readout, 1 - P_readout],
    [1 - P_readout, P_readout]
])
```
- using the generative model to create some data
```python
measured_states = []  # array to hold the measured states (even or odd), this data is available
true_states = []  # array to hold the true states (even or odd), this data is hidden

repeats = 1000  # how many initialisation, measurement, measurement, ... sequences to perform
measurements = 20 # how many measurements in the above sequence.
```
- generating the data

- appending the data to the arrays

- making the data arrays (python lists) into numpy arrays for convenience

- plotting the generated data

- %%  coming up with heuristic priors for the parameters

- the probability the first measurement is even is
- P(first measurement even) = P(init even) P(measure even | even) + P(init odd) P(measure even | odd)
- if the initialisation and readout fidelities are any good then
- P(init even) P(measure even | even) >> P(init odd) P(measure even | odd)
- P(first measurement even) approx_eq P(init even) P(measure even | even)
- very heuristic we can assume P(init even) = P(measure even | even) therefore
- P(first measurement even) = P(init even)^2 = P(measure even | even)^2

- the probability of measuring even consecutively for N measurements from initialisation is approximately
- (neglecting readout errors):
- P(no transitions) = P(init even) * (1 - P(even to odd))^N
- P(even to odd) = 1 - (P(no transitions) / P(init even)) ^ (1 / N)


- if one considers a transition matrix of the form
- [[1 - P_eo, P_eo],
-  [P_oe, 1 - P_oe]]
- the stead state is [P_oe, P_eo] / (P_eo + P_oe). So the ratio of the occurrence of the even and odd state is
- P_oe / P_eo. So the probability a measurement is odd in the stead-state P_odd_ss = P_eo / (P_eo + P_oe) and
- P_oe = (1 / P_odd_ss - 1) * P_eo.
- We assume that the statistics of the last measurement of each sequence is in the stead state


- %%  fitting hidden markov models to the data


- in principle there could be a lot of data, which would be computationally expensive to fit to. So we randomly select
- a subset of the data to fit to. The variable "sequences_in_subset" sets how many initialisation, measurement, ...
- sequences are included in the subset. For each of these subsets we fit a hidden markov model with a random
- initialisation informed by our priors.

- an array to inform the fit about the shape of the data, aka how many measurements are in each sequence
    - creating the hidden markov model
    - setting the initial parameters of our hmm model somewhat randomly according to our priors on the parameters

    - creating the random subset of the data
    - fitting the model to the subset of the data
    - storing the score of the fitted model, evaluated over the whole dataset


- %% plotting the distribution of the parameters for the maximum likelihood fit. The variation arises from two sources
- 1. the randomness of the initial conditions resulting in the optimiser converging to different locations
- 2. the randomness of the subset of data

- extracting the fit parameters and associated score for each of the fitted models


- finding the model which fits the data best

- fitting the model to the complete dataset just incase it changes the minimum slightly

- taking the parameters out of matrix form


- actually plotting



