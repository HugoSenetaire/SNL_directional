import numpy as np
from numpy.matlib import repmat
from scipy.linalg import null_space

from .sampling_marginal_t import rand_t_marginal
from .uniform import rand_uniform_hypersphere


def rand_von_mises_fisher(mu, kappa, N=1):
    """
    rand_von_mises_fisher(mu,kappa,N=1)
    ===================================

    Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

    INPUT:

        * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
        * kappa (float) - concentration.
        * N (int) - Number of samples.

    OUTPUT:

        * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
        with mean direction mu and concentration kappa.
    """

    # Check that mu is a unit vector
    eps = 10 ** (-4)  # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # Dimension p
    p = len(mu)

    # Make sure that mu has a shape of px1
    mu = np.reshape(mu, (p, 1))

    # Array to store samples
    samples = np.zeros((N, p))

    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal(kappa, p, N)

    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere(N, p - 1)

    # von-Mises-Fisher samples Nxp

    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:, [0]] = t

    # Component orthogonal to mu (Nx(p-1))
    samples[:, 1:] = repmat(np.sqrt(1 - t**2), 1, p - 1) * xi

    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples
