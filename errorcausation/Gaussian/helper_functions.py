import dataclasses

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataclasses import dataclass


def array_to_complex(X):
    return X[..., 0] + 1j * X[..., 1]


@dataclasses.dataclass
class ReadoutCorrections:
    I_correction: float
    Q_correction: float
    phase_correction: float
    decision_boundary: float
    f_ground: float
    f_excited: float


def apply_correction(I, Q, corrections):
    z = I + 1j * Q
    z = z - (corrections.I_correction + 1j * corrections.Q_correction)
    z = z * np.exp(1j * corrections.phase_correction)
    return z.real, z.imag


def readout_corrections(model):
    means = model.means_
    stds = model.covar_to_std()
    startprob = model.startprob_

    ground_id = np.argmax(startprob)
    excited_id = np.argmin(startprob)

    ground_mean = array_to_complex(means[ground_id])
    excited_mean = array_to_complex(means[excited_id])

    partically_correct_exited_mean = excited_mean - ground_mean
    phase_correction = -np.angle(partically_correct_exited_mean)

    fully_corrected_excited_mean = partically_correct_exited_mean * np.exp(1j * phase_correction)

    ground_std, excited_std = stds[ground_id], stds[excited_id]
    coefficients = np.array([
        1 - (ground_std / excited_std) ** 2,
        -2 * fully_corrected_excited_mean.real,
        fully_corrected_excited_mean.real ** 2]
    )
    roots = np.roots(coefficients)
    applicable_roots = roots[np.logical_and(roots > 0, roots < fully_corrected_excited_mean.real)]
    assert applicable_roots.__len__() == 1, f"Found {applicable_roots.__len__()} applicable roots: {applicable_roots} there should be exactly one."
    decision_boundary = applicable_roots[0]


    ground_dist, excited_dist = norm(0, ground_std), norm(fully_corrected_excited_mean.real, excited_std)

    f_ground = ground_dist.cdf(decision_boundary)
    f_excited = 1 - excited_dist.cdf(decision_boundary)

    return ReadoutCorrections(
        I_correction=ground_mean.real,
        Q_correction=ground_mean.imag,
        phase_correction=phase_correction,
        decision_boundary=decision_boundary,
        f_ground=f_ground,
        f_excited=f_excited
    )


def readout_fidelity(model, X, plot=False):
    predictions = model.predict(X)
    corrections = readout_corrections(model)

    I = X[..., 0].flatten()
    Q = X[..., 1].flatten()

    I_corrected, Q_corrected = apply_correction(I, Q, corrections)

    I_given_ground = I[predictions.flatten() == 0]
    I_given_excited = I[predictions.flatten() == 1]
    Q_given_ground = Q[predictions.flatten() == 0]
    Q_given_excited = Q[predictions.flatten() == 1]

    I_corrected_given_ground = I_corrected[predictions.flatten() == 0]
    I_corrected_given_excited = I_corrected[predictions.flatten() == 1]
    Q_corrected_given_ground = Q_corrected[predictions.flatten() == 0]
    Q_corrected_given_excited = Q_corrected[predictions.flatten() == 1]

    f_ground = 1 - np.mean(I_corrected_given_ground > corrections.decision_boundary)
    f_excited = 1 - np.mean(I_corrected_given_excited < corrections.decision_boundary)

    if f_ground < 0.5 and f_excited < 0.5:
        f_ground = 1 - f_ground
        f_excited = 1 - f_excited

    print(f"Frequentest f_ground, f_excited = {f_excited:.3f}, {f_ground:.3f}")
    print(f"Bayesian f_ground, f_excited = {corrections.f_ground:.3f}, {corrections.f_excited:.3f}")

    if plot:
        IQ_range = max([I.max() - I.min(), Q.max() - Q.min()])
        I_bins = np.linspace(I.mean() - IQ_range / 2, I.mean() + IQ_range / 2, 40)
        Q_bins = np.linspace(Q.mean() - IQ_range / 2, Q.mean() + IQ_range / 2, 40)

        corrected_IQ_range = max([I_corrected.max() - I_corrected.min(), Q_corrected.max() - Q_corrected.min()])
        I_corrected_bins = np.linspace(I_corrected.mean() - corrected_IQ_range / 2,
                                       I_corrected.mean() + corrected_IQ_range / 2, 40)
        Q_corrected_bins = np.linspace(Q_corrected.mean() - corrected_IQ_range / 2,
                                       Q_corrected.mean() + corrected_IQ_range / 2, 40)

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        fig.set_size_inches(5, 5)
        ax[0, 0].hist2d(I, Q, cmap='hot', bins=[I_bins, Q_bins])

        for i in range(model.n_components):
            std = model.covar_to_std()[i]
            ax[0, 0].add_artist(plt.Circle(model.means_[i], 2 * std, color='b', fill=False))
        ax[0, 0].set_xlabel('I')
        ax[0, 0].set_ylabel('Q')
        ax[0, 0].set_xlim(I_bins[0], I_bins[-1])
        ax[0, 0].set_ylim(Q_bins[0], Q_bins[-1])
        ax[0, 0].set_aspect('equal')

        ax[0, 1].scatter(I_given_ground, Q_given_ground, s=0.1, label='ground')
        ax[0, 1].scatter(I_given_excited, Q_given_excited, s=0.1, label='excited')
        ax[0, 1].set_xlabel('I')
        ax[0, 1].set_ylabel('Q')
        ax[0, 1].set_xlim(I_bins[0], I_bins[-1])
        ax[0, 1].set_ylim(Q_bins[0], Q_bins[-1])
        ax[0, 1].set_aspect('equal')

        ax[1, 0].hist2d(I_corrected, Q_corrected, cmap='hot', bins=[I_corrected_bins, Q_corrected_bins])

        for i in range(model.n_components):
            std = model.covar_to_std()[i]
            rotated_mean = apply_correction(*model.means_[i], corrections)

            ax[1, 0].add_artist(plt.Circle(rotated_mean, 2 * std, color='b', fill=False))
        ax[1, 0].set_xlabel('I')
        ax[1, 0].set_ylabel('Q')
        ax[1, 0].set_xlim(I_corrected_bins[0], I_corrected_bins[-1])
        ax[1, 0].set_ylim(Q_corrected_bins[0], Q_corrected_bins[-1])
        ax[1, 0].set_aspect('equal')
        ax[1, 0].axvline(corrections.decision_boundary, color='w', linestyle='--')

        ax[1, 1].scatter(I_corrected_given_ground, Q_corrected_given_ground, s=0.1, label='ground')
        ax[1, 1].scatter(I_corrected_given_excited, Q_corrected_given_excited, s=0.1, label='excited')
        ax[1, 1].set_xlabel('I')
        ax[1, 1].set_ylabel('Q')
        ax[1, 1].set_xlim(I_corrected_bins[0], I_corrected_bins[-1])
        ax[1, 1].set_ylim(Q_corrected_bins[0], Q_corrected_bins[-1])
        ax[1, 1].set_aspect('equal')
        ax[1, 1].axvline(corrections.decision_boundary, color='k', linestyle='--')

        fig.tight_layout()
