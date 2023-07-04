import numpy as np
from qm.qua import *
from qualang_tools.loops import qua_arange
from qm.qua import Math
from qm.qua import Util
from qm.qua import Cast

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


def create_forward_program(observations, model, normalise=True, overflow_protect=True, if_incorperate_Q=True):
    # enforcing the types of the inputs

    startprob = np.array([0.5, 0.5])
    transmat = model.transmat_
    means = model.means_
    covs = model.covars_

    observations = observations.squeeze()
    # storing the number of observations and hidden states
    number_of_observations, number_of_hidden_states = observations.shape[0], 2

    I_std_inv = 1 / np.sqrt(np.diag(covs[0, ...]))
    Q_std_inv = 1 / np.sqrt(np.diag(covs[1, ...]))

    I_std_inv_int = np.rint(I_std_inv).astype(int)
    Q_std_inv_int = np.rint(Q_std_inv).astype(int)

    I_std_inv_frac = (I_std_inv / I_std_inv_int)
    Q_std_inv_frac = (Q_std_inv / Q_std_inv_int)

    with program() as forward:
        alpha_0_stream = declare_stream()
        alpha_1_stream = declare_stream()

        # index variables
        n = declare(int, value=0)

        # temporary variables
        sum = declare(fixed, value=0.)
        temp_array = declare(fixed, value=[0., 4.])
        temp = declare(fixed, value=0.)

        I_temp = declare(fixed, value=0.)
        Q_temp = declare(fixed, value=0.)

        I_means = QuaArray1d(means[:, 0], fixed)
        Q_means = QuaArray1d(means[:, 1], fixed)

        I_std_inv_frac = QuaArray1d(I_std_inv_frac, fixed)
        Q_std_inv_frac = QuaArray1d(Q_std_inv_frac, fixed)

        I_std_inv_int = QuaArray1d(I_std_inv_int, int)
        Q_std_inv_int = QuaArray1d(Q_std_inv_int, int)

        I_observations = QuaArray1d(observations[:, 0], fixed)
        Q_observations = QuaArray1d(observations[:, 1], fixed)
        startprob = QuaArray1d(startprob, fixed)

        alpha = QuaArray2d(np.zeros((number_of_observations, number_of_hidden_states)), fixed)
        transmat = QuaArray2d(transmat, fixed)

        def normal(x, mu, inv_cov_int, inv_cov_frac):
            if overflow_protect:
                assign(temp_array[0], Cast.mul_fixed_by_int(Math.abs(x - mu) * inv_cov_frac, inv_cov_int))
                assign(temp, Math.min(temp_array))
            else:
                assign(temp, Cast.mul_fixed_by_int(Math.abs(x - mu) * inv_cov_frac, inv_cov_int))
            return Math.exp(-0.5 * temp * temp)

        for k in range(number_of_hidden_states):
            assign(I_temp, normal(I_observations[0], I_means[k], I_std_inv_int[k], I_std_inv_frac[k]))

            if if_incorperate_Q:
                assign(Q_temp, normal(Q_observations[0], Q_means[k], Q_std_inv_int[k], Q_std_inv_frac[k]))
            else:
                assign(Q_temp, 1)
            assign(alpha[0, k], startprob[k] * I_temp * Q_temp)

        if normalise:
            # normalizing the first alpha values
            assign(sum, alpha[0, 0] + alpha[0, 1])
            assign(alpha[0, 0], alpha[0, 0] / sum)
            assign(alpha[0, 1], alpha[0, 1] / sum)

        save(alpha[0, 0], alpha_0_stream)
        save(alpha[0, 1], alpha_1_stream)

        with for_(*qua_arange(n, 1, number_of_observations, 1)):
            for m in range(number_of_hidden_states):
                assign(sum, 0.)
                for k in range(number_of_hidden_states):
                    assign(sum, sum + alpha[n - 1, k] * transmat[k, m])
                assign(I_temp, normal(I_observations[n], I_means[m], I_std_inv_int[m], I_std_inv_frac[m]))
                if if_incorperate_Q:
                    assign(Q_temp, normal(Q_observations[0], Q_means[k], Q_std_inv_int[k], Q_std_inv_frac[k]))
                else:
                    assign(Q_temp, 1)

                assign(alpha[n, m], sum * I_temp * Q_temp)

            if normalise:
                # normalizing the alpha values
                assign(sum, alpha[n, 0] + alpha[n, 1])
                assign(alpha[n, 0], alpha[n, 0] / sum)
                assign(alpha[n, 1], alpha[n, 1] / sum)

            # saving the alpha values to the streams
            save(alpha[n, 0], alpha_0_stream)
            save(alpha[n, 1], alpha_1_stream)

        with stream_processing():
            alpha_0_stream.buffer(number_of_observations).save("p0")
            alpha_1_stream.buffer(number_of_observations).save("p1")
            alpha_0_stream.timestamps().buffer(number_of_observations).save("p0_timestamps")
            alpha_1_stream.timestamps().buffer(number_of_observations).save("p1_timestamps")

    return forward
