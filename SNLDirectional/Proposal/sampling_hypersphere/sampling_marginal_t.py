import numpy as np


def rand_t_marginal(kappa, p, N=1):
    """
    rand_t_marginal(kappa,p,N=1)
    ============================

    Samples the marginal distribution of t using rejection sampling of Wood [3].

    INPUT:

        * kappa (float) - concentration
        * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
            - p = 2 for the unit circle $\mathbb{S}^{1}$
            - p = 3 for the unit sphere $\mathbb{S}^{2}$
        Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
        samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
        * N (int) - number of samples

    OUTPUT:

        * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    if (p <= 0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)

    samples = np.zeros((N, 1))

    # Loop over number of samples
    for i in range(N):

        # Continue unil you have an acceptable sample
        while True:

            # Sample Beta distribution
            Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)

            # Sample Uniform distribution
            U = np.random.uniform(low=0.0, high=1.0)

            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):

                # Accept sample
                samples[i] = W
                break

    return samples
