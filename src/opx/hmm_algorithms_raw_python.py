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
    F = np.zeros((N, M))
    F[0, :] = startprob * emmisonprob[:, observations[0]]
    for n in range(1, N):
        for m in range(M):
            F[n, m] = np.sum(F[n-1, :] * transmat[:, m]) * emmisonprob[m, observations[n]]
    return F / np.sum(F, axis=1).reshape(-1, 1)

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

