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


def forward(observations, startprob, transmat, means, covs):
    """Forward algorithm for HMMs.
        O: observation sequence
        S: set of states
        Pi: initial state probabilities
        Tm: transition matrix
        Em: emission matrix
        """
    observations = observations.squeeze()

    I_observations = observations[:, 0]
    Q_observations = observations[:, 1]

    startprob = startprob.squeeze()
    transmat = transmat.squeeze()
    means = means.squeeze()
    covs = covs.squeeze()

    I_means = means[:, 0]
    Q_means = means[:, 1]

    I_covs = np.diag(covs[0, ...])
    Q_covs = np.diag(covs[1, ...])

    N, M = len(observations), 2
    alpha = np.zeros((N, M))
    alpha[0, :] = startprob * np.exp(-0.5 * (I_observations[0] - I_means)**2 / I_covs)
    alpha[0, :] /= np.sum(alpha[0, :])
    for n in range(1, N):
        for m in range(M):
            I_obs = np.exp(-0.5 * (I_observations[n] - I_means[m])**2 / I_covs[m])
            Q_obs = np.exp(-0.5 * (Q_observations[n] - Q_means[m])**2 / Q_covs[m])
            alpha[n, m] = np.sum(alpha[n-1, :] * transmat[:, m]) * I_obs * Q_obs
        alpha[n, :] /= np.sum(alpha[n, :])
    return alpha

def backward(O, S, Pi, Tm, Em):
    """Backward algorithm for HMMs.
        O: observation sequence
        S: set of states
        Pi: initial state probabilities
        Tm: transition matrix
        Em: emission matrix
        """
    N = len(O)
    M = len(S)
    B = np.zeros((N, M))
    B[N-1] = 1
    for n in range(N-2, -1, -1):
        for m in range(M):
            B[n, m] = np.sum(B[n+1] * Tm[m, :] * Em[:, O[n+1]])
    return B

def baum_welch(O, S, Pi, Tm, Em, N=100):
    """Baum-Welch algorithm for HMMs.
        O: observation sequence
        S: set of states
        Pi: initial state probabilities
        Tm: transition matrix
        Em: emission matrix
        N: number of iterations
        """
    M = len(S)
    for n in range(N):
        F = forward(O, S, Pi, Tm, Em)
        B = backward(O, S, Pi, Tm, Em)
        P = F * B
        P = P / np.sum(P, axis=1)[:, np.newaxis]
        Pi = P[0]
        Tm = np.zeros((M, M), dtype=int)
        for m in range(M):
            Tm[m] = np.sum(P[:-1, m] * Tm[:, m] * Em[:, O[1:]], axis=0) / np.sum(P[:-1, m])
        Em = np.zeros((M, M), dtype=int)
        for m in range(M):
            Em[m] = np.sum(P[:, m] * O) / np.sum(P[:, m])
    return Pi, Tm, Em

