from scipy.sparse import eye
from scipy.sparse.linalg import splu

import scipy.sparse
from scipy.sparse import coo_array, csr_array, eye
import numpy as np
from numpy.typing import NDArray
import random

def Rademacher_vec(len):
    return 2.0 * np.random.randint(2, size=(len,1)) - 1.0


def chebyshev_coeffs(delta: float, n_degree: int) -> NDArray[np.float64]:
    """Compute the coefficients for the chebyshev approximation of log(1 - ((1-2*delta)*x + 1))/2 )

    Args:
        delta (float): delta
        n_degree (int): degree of polynomial approximation

    Returns:
        NDArray[np.float64]: the coefficients
    """

    # alias for shortness
    n = n_degree

    k = np.arange(0, n + 1)
    xk = np.cos(np.pi * (k + 1 / 2) / (n + 1))
    fxk = np.log(1.0 - ((1.0 - 2.0 * delta) * xk + 1.0) / 2.0)
    # fxk = np.log(1 - xk)

    # The coefficients
    c = np.zeros(n_degree + 1)

    Txk = np.zeros((n + 1) * (n + 1)).reshape((n + 1), (n + 1))
    Txk[0] = np.ones(n + 1)
    Txk[1] = xk
    c[0] = 1.0 / (n + 1) * np.sum(fxk * Txk[0])
    for i in range(1, n):
        Txk[i + 1] = 2.0 * xk * Txk[i] - Txk[i - 1]
        c[i] = 2 / (n + 1) * np.sum(fxk * Txk[i])
    c[n] = 2.0 / (n + 1.0) * np.sum(fxk * Txk[n])

    return c


def logdet(matrix: csr_array, n_sample: int, n_degree: int, delta: float) -> float:
    """Computes the log-determinant using the chebyshev approximation

    Args:
        matrix (csr_array): positive definite determinant
        n_sample (int): number of random samples
        n_degree (int): polynomial degree of chebyshev approximation
        delta (float): the eigenvalues of matrix have to be in the interval [delta, 1-delta]

    Returns:
        float: the log determinant
    """

    d = matrix.shape[0]
    I = scipy.sparse.eye(d, d)

    A = (2.0 * matrix - I) / (2.0 * delta - 1.0)

    Gamma = 0.0

    c = chebyshev_coeffs(delta, n_degree)

    # Calculate Log Determinant via Monte-Carlo Method
    for i in range(n_sample):
        # print("Iteration: ", i)
        v = Rademacher_vec(d)
        u = c[0] * v
        if n_degree > 1:
            w0 = v
            w1 = A@v
            u = u + c[1] * w1
            for j in range(2, n_degree + 1):
                w2 = 2.0 * A@w1 - w0
                u = u + c[j] * w2
                w0 = w1
                w1 = w2
        Gamma = Gamma + v.T@u / n_sample

    return Gamma[0,0]
