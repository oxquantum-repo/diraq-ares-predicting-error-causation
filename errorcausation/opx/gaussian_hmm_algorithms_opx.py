import numpy as np
from qm.qua import *
from qualang_tools.loops import qua_arange
from qm.qua import Math

from errorcausation.helperfunctions.arraymanipulations import ravel_index

class QuaArray1d:
    def __init__(self, array, type):
        assert len(array.shape) == 1, "The qua array must be 1 dimensional"
        self.shape = array.shape[0]
        self.qua_array = declare(type, value=array.tolist())

    def index(self, i):
        return i

    def get(self, i):
        return self.qua_array[self.index(i)]

    def set(self, i, value):
        assign(self.qua_array[self.index(i)], value)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

class QuaArray2d:
    def __init__(self, array, type):
        assert len(array.shape) == 2, "The qua array must be 2 dimensional"
        self.shape = array.shape
        self.qua_array = declare(type, value=array.flatten().tolist())

    def index(self, i, j):
        return i * self.shape[0] + j

    def get(self, i, j):
        return self.qua_array[self.index(i, j)]

    def set(self, i, j, value):
        assign(self.qua_array[self.index(i, j)], value)

    def __getitem__(self, key):
        return self.get(*key)

    def __setitem__(self, key, value):
        self.set(*key, value)


def create_forward_program(observations: np.array, startprob: np.array, transmat: np.array, means: np.array, covs: np.array):
    # enforcing the types of the inputs

    observations = observations.squeeze()
    # storing the number of observations and hidden states
    number_of_observations, number_of_hidden_states = observations.shape[0], 2

    I_std_inv = 1 / np.sqrt(np.diag(covs[0, ...]))
    Q_std_inv = 1 / np.sqrt(np.diag(covs[1, ...]))

    I_covs_inv_int = np.rint(I_std_inv).astype(int)
    Q_covs_inv_int = np.rint(Q_std_inv).astype(int)

    I_covs_inv_frac = (I_std_inv / I_covs_inv_int)
    Q_covs_inv_frac = (Q_std_inv / Q_covs_inv_int)

    with program() as forward:
        alpha_0_stream = declare_stream()
        alpha_1_stream = declare_stream()

        # index variables
        n = declare(int, value=0)
        m = declare(int, value=0)
        k = declare(int, value=0)

        # temporary variables
        sum = declare(fixed, value=0.)
        scalar_temp = declare(fixed, value=0.)

        temp = QuaArray1d(np.zeros(2), fixed)

        I_temp = declare(fixed, value=0.)
        Q_temp = declare(fixed, value=0.)

        I_means = QuaArray1d(means[:, 0], fixed)
        Q_means = QuaArray1d(means[:, 0], fixed)

        I_std_inv = QuaArray1d(I_std_inv, fixed)
        Q_std_inv = QuaArray1d(Q_std_inv, fixed)

        I_observations = QuaArray1d(observations[:, 0], fixed)
        Q_observations = QuaArray1d(observations[:, 1], fixed)
        startprob = QuaArray1d(startprob, fixed)

        alpha = QuaArray2d(np.zeros((number_of_observations, number_of_hidden_states)), fixed)
        transmat = QuaArray2d(transmat, fixed)
        def normalize(x):
            assign(sum, x[0] + x[1])
            assign(x[0], x[0] / sum)
            assign(x[1], x[1] / sum)

        def normal(x, mu, inv_cov):
            assign(scalar_temp, (x - mu) * inv_cov)
            return Math.exp(-0.5 * scalar_temp * scalar_temp)

        with for_(*qua_arange(k, 0, number_of_hidden_states, 1)):
            assign(I_temp, normal(I_observations[0], I_means[k], I_std_inv[k]))
            assign(Q_temp,normal(Q_observations[0], Q_means[k], Q_std_inv[k]))
            assign(temp[k], startprob[k] * I_temp * Q_temp)

        normalize(temp)
        assign(alpha[0, 0], temp[0])
        assign(alpha[0, 1], temp[1])
        # saving the first alpha values to the streams
        save(temp[0], alpha_0_stream)
        save(temp[1], alpha_1_stream)

        with for_(*qua_arange(n, 1, number_of_observations, 1)):
            with for_(*qua_arange(m, 0, number_of_hidden_states, 1)):
                assign(sum, 0.)
                with for_(*qua_arange(k, 0, number_of_hidden_states, 1)):
                    assign(sum, sum + alpha[n - 1, k] * transmat[k, m])

                assign(I_temp, normal(I_observations[n], I_means[m], I_std_inv[m]))
                assign(Q_temp, normal(Q_observations[n], Q_means[m], Q_std_inv[m]))
                assign(temp[m], sum * I_temp * Q_temp)

            normalize(temp)
            # rescaling alpha_0 and alpha_1 such that they sum to one and saving them to the alpha array
            assign(alpha[n, 0], temp[0])
            assign(alpha[n, 1], temp[1])

            # saving the alpha values to the streams
            save(temp[0], alpha_0_stream)
            save(temp[1], alpha_1_stream)

        with stream_processing():
            alpha_0_stream.buffer(number_of_observations).save("p0")
            alpha_1_stream.buffer(number_of_observations).save("p1")
            alpha_0_stream.timestamps().buffer(number_of_observations).save("p0_timestamps")
            alpha_1_stream.timestamps().buffer(number_of_observations).save("p1_timestamps")

    return forward
