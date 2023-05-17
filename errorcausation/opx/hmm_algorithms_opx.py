from qm.qua import *
from qualang_tools.loops import qua_arange
from qm.qua import Math

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

        def normalize(x):
            assign(sum, x[0] + x[1])
            assign(x[0], x[0] / sum)
            assign(x[1], x[1] / sum)

        # saving the parameters of the hhm as qua arrays
        observations = declare(int, value=observations)
        transmat = declare(fixed, value=transmat)
        emmissionprob = declare(fixed, value=emmissionprob)

        # the alpha array to store the forward probabilities
        alpha = declare(fixed, size=number_of_hidden_states * number_of_observations)
        assign(temp[0], startprob[0] * emmissionprob[index_emmissionprob(0, observations[0])])
        assign(temp[1], startprob[1] * emmissionprob[index_emmissionprob(1, observations[0])])

        normalize(temp)
        assign(alpha[index_alpha(0, 0)], temp[0])
        assign(alpha[index_alpha(0, 1)], temp[1])
        # saving the first alpha values to the streams
        save(temp[0], alpha_0_stream)
        save(temp[1], alpha_1_stream)

        with for_(*qua_arange(n, 1, number_of_observations, 1)):
            with for_(*qua_arange(m, 0, number_of_hidden_states, 1)):
                assign(sum, 0.)
                with for_(*qua_arange(k, 0, number_of_hidden_states, 1)):
                    assign(sum, sum + alpha[index_alpha(n - 1, k)] * transmat[index_transmat(k, m)])
                assign(temp[m], sum * emmissionprob[index_emmissionprob(m, observations[n])])

            normalize(temp)
            # rescaling alpha_0 and alpha_1 such that they sum to one and saving them to the alpha array
            assign(alpha[index_alpha(n, 0)], temp[0])
            assign(alpha[index_alpha(n, 1)], temp[1])

            # saving the alpha values to the streams
            save(temp[0], alpha_0_stream)
            save(temp[1], alpha_1_stream)

        with stream_processing():
            alpha_0_stream.buffer(number_of_observations).save("p0")
            alpha_1_stream.buffer(number_of_observations).save("p1")




    return forward
