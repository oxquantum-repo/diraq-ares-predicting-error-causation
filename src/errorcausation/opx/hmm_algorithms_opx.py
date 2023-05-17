from qm.qua import *
from qualang_tools.loops import qua_arange

from errorcausation.helperfunctions.arraymanipulations import ravel_index


def create_forward_program(observations: np.array, startprob: np.array, transmat: np.array, emissionprob: np.array):
    # enforcing the types of the inputs
    observations = observations.astype(int).squeeze().tolist()
    startprob = startprob.flatten().tolist()
    transmat = transmat.flatten().tolist()
    emmissionprob = emissionprob.flatten().tolist()

    # storing the number of observations and hidden states
    number_of_observations, number_of_hidden_states = len(observations), 2

    # defining the index functions for convenience
    index_alpha = lambda i, j: ravel_index(i, j, number_of_hidden_states)
    index_transmat = lambda i, j: ravel_index(i, j, number_of_hidden_states)
    index_emmissionprob = lambda i, j: ravel_index(i, j, number_of_hidden_states)

    with program() as forward:
        alpha_0_stream = declare_stream()
        alpha_1_stream = declare_stream()

        # index variables
        n = declare(int, value=0)
        m = declare(int, value=0)
        k = declare(int, value=0)

        # temporary variables
        sum = declare(fixed, value=0.)
        temp = declare(fixed, value=[0., 0.])
        recip = declare(fixed, value=0.)

        # saving the parameters of the hhm as qua arrays
        observations = declare(int, value=observations)
        transmat = declare(fixed, value=transmat)
        emmissionprob = declare(fixed, value=emmissionprob)

        # the alpha array to store the forward probabilities
        alpha = declare(fixed, size=number_of_hidden_states * number_of_observations)
        assign(alpha[index_alpha(0, 0)], startprob[0] * emmissionprob[index_emmissionprob(0, observations[0])])
        assign(alpha[index_alpha(0, 1)], startprob[1] * emmissionprob[index_emmissionprob(1, observations[0])])

        # saving the first alpha values to the streams
        save(alpha[index_alpha(n, 0)], alpha_0_stream)
        save(alpha[index_alpha(n, 1)], alpha_1_stream)

        with for_(*qua_arange(n, 1, number_of_observations, 1)):
            with for_(*qua_arange(m, 0, number_of_hidden_states, 1)):
                assign(sum, 0.)
                with for_(*qua_arange(k, 0, number_of_hidden_states, 1)):
                    assign(sum, sum + alpha[index_alpha(n - 1, k)] * transmat[index_transmat(k, m)])
                assign(temp[m], sum * emmissionprob[index_emmissionprob(m, observations[n])])

            # saving the reciprocal of the sum of the two alpha values, as it is used twice and division is expensive
            assign(recip, 1. / (temp[0] + temp[1]))
            # rescaling alpha_0 and alpha_1 such that they sum to one and saving them to the alpha array
            assign(alpha[index_alpha(n, 0)], temp[0] * recip)
            assign(alpha[index_alpha(n, 1)], temp[1] * recip)

            # saving the alpha values to the streams
            save(alpha[index_alpha(n, 0)], alpha_0_stream)
            save(alpha[index_alpha(n, 1)], alpha_1_stream)

        with stream_processing():
            # # as when it is done on the FPGA there is the possibility of fixed point errors
            (alpha_0_stream / (alpha_0_stream + alpha_1_stream)).buffer(number_of_observations).save("p0")
            (alpha_1_stream / (alpha_0_stream + alpha_1_stream)).buffer(number_of_observations).save("p1")



    return forward
