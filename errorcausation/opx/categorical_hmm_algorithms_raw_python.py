import numpy as np
def viterbi(O, S, Pi, Tm, Em):
    """Viterbi algorithm for HMMs.
    O: observation sequence
    S: set of states
    Pi: initial state probabilities
    Tm: transition matrix
    Em: emission matrix
    """
    N = len(O)
    M = len(S)
    T = np.zeros((N, M))
    T[0] = Pi * Em[:, O[0]]
    P = np.zeros((N, M))
    for n in range(1, N):
        for m in range(M):
            T[n, m] = np.max(T[n-1] * Tm[:, m]) * Em[m, O[n]]
            P[n, m] = np.argmax(T[n-1] * Tm[:, m])
    q = np.zeros(N, dtype=int)
    q[N-1] = np.argmax(T[N-1])
    for n in range(N-2, -1, -1):
        q[n] = P[n+1, q[n+1]]
    return q

def forward(observations, startprob, transmat, emmisonprob):
    """Forward algorithm for HMMs.
        O: observation sequence
        S: set of states
        Pi: initial state probabilities
        Tm: transition matrix
        Em: emission matrix
        """
    observations = observations.astype(int).squeeze()
    startprob = startprob.squeeze()
    transmat = transmat.squeeze()
    emmisonprob = emmisonprob.squeeze()

    N, M = len(observations), 2
    alpha = np.zeros((N, M))
    alpha[0, :] = startprob * emmisonprob[:, observations[0]]
    alpha[0, :] /= np.sum(alpha[0, :])
    for n in range(1, N):
        for m in range(M):
            alpha[n, m] = np.sum(alpha[n-1, :] * transmat[:, m]) * emmisonprob[m, observations[n]]
        alpha[n, :] /= np.sum(alpha[n, :])
    return alpha

def backward(observations, startprob, transmat, emmisonprob):
    """
    Backward algorithm for HMMs.
    """
    observations = observations.astype(int).squeeze()
    startprob = startprob.squeeze()
    transmat = transmat.squeeze()
    emmisonprob = emmisonprob.squeeze()

    N, M = len(observations), 2
    beta = np.zeros((N, M))
    beta[N-1, :] = 1
    for n in range(N-2, -1, -1):
        beta[n, :] = np.sum(beta[n+1, :] * transmat * emmisonprob[:, observations[n+1]], axis=1)
        beta[n, :] /= np.sum(beta[n, :])
    return beta



def baum_welch(observations, startprob, transmat, emmisonprob, n_iter=10):
    """
    Baum-Welch algorithm for HMMs.
    """

    observations = observations.astype(int).squeeze()
    startprob = startprob.squeeze()
    transmat = transmat.squeeze()
    emmisonprob = emmisonprob.squeeze()

    N, M = len(observations), 2

    for _ in range(n_iter):

        alpha = forward(observations, startprob, transmat, emmisonprob)
        beta = backward(observations, startprob, transmat, emmisonprob)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        xi = np.zeros((N-1, M, M))
        for n in range(N-1):
            for i in range(M):
                for j in range(M):
                    xi[n, i, j] = alpha[n, i] * transmat[i, j] * emmisonprob[j, observations[n+1]] * beta[n+1, j]
        xi /= np.sum(xi, axis=(1, 2))[:, np.newaxis, np.newaxis]
        startprob[...] = gamma[0]
        transmat[...] = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
        for k in range(M):
            for l in range(M):
                transmat[k, l] /= np.sum(transmat[k, :])
        for k in range(M):
            for l in range(M):
                emmisonprob[k, l] = np.sum(gamma[:, k] * (observations == l)) / np.sum(gamma[:, k])

    return startprob, transmat, emmisonprob

