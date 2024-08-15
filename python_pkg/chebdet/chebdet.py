import scipy.sparse
from scipy.sparse import csr_array
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Union


def rademacher_vec(len: int) -> NDArray[np.float64]:
    return 2.0 * np.random.randint(2, size=(len, 1)) - 1.0


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
    xk = np.cos(np.pi * (k + 0.5) / (n + 1))
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
        c[i] = 2.0 / (n + 1.0) * np.sum(fxk * Txk[i])
    c[n] = 2.0 / (n + 1.0) * np.sum(fxk * Txk[n])

    return c


def log_det_positive_definite_unit_interval(
    matrix: Union[csr_array, NDArray],
    n_sample: int,
    n_degree: int,
    delta: float,
    eigenvalues_deflate: Optional[List[float]] = None,
    eigenvectors_deflate: Optional[List[NDArray[np.float64]]] = None,
) -> float:
    """Computes the log-determinant of a positive definite matrix with eigenvalues smaller than one.

    Args:
        matrix (csr_array | NDArray): positive definite determinant
        n_sample (int): number of random samples
        n_degree (int): polynomial degree of chebyshev approximation
        delta (float): the eigenvalues of matrix have to be in the interval [delta, 1-delta]
        eigenvalues_deflate (Optional[List[float]]): The list of eigenvalues to remove
        eigenvectors_deflate (Optional[List[NDArray[np.float64]]]): The list of eigenvectors to remove


    Returns:
        float: the log determinant
    """

    deflate = not eigenvalues_deflate is None and not eigenvectors_deflate is None
    if deflate:
        if len(eigenvalues_deflate) != len(eigenvectors_deflate):
            raise Exception(
                "Eigenvectors deflate and eigenvalues deflate need to have the same length"
            )

        # For the deflation use a target that's within the [delta, 1-delta] interval
        deflate_target = 0.5
        n_deflate = len(eigenvalues_deflate)

    d = matrix.shape[0]
    I = scipy.sparse.eye(d, d)

    scale = 2.0 / (2.0 * delta - 1.0)

    A = scale * (matrix - 0.5 * I)

    Gamma = 0.0

    c = chebyshev_coeffs(delta, n_degree)

    def mult_A(lhs: NDArray[np.float64]):
        res = A @ lhs

        if deflate:
            for eval, evec in zip(eigenvalues_deflate, eigenvectors_deflate):
                res += scale * (deflate_target - eval) * evec * (evec.T @ lhs)[0, 0]

        return res

    # Calculate Log Determinant via Monte-Carlo Method
    for i in range(n_sample):
        v = rademacher_vec(d)
        u = c[0] * v
        if n_degree > 1:
            w0 = v
            w1 = mult_A(v)
            u = u + c[1] * w1
            for j in range(2, n_degree + 1):
                w2 = 2.0 * mult_A(w1) - w0
                u = u + c[j] * w2
                w0 = w1
                w1 = w2
        Gamma = Gamma + v.T @ u / n_sample

    res = Gamma[0, 0]

    if deflate:
        for i in range(n_deflate):
            res -= np.log(deflate_target)

    return res


def log_det_positive_definite(
    matrix: Union[csr_array, NDArray],
    n_sample: int,
    n_degree: int,
    sigma_max: float,
    eigenvalues_deflate: Optional[List[float]] = None,
    eigenvectors_deflate: Optional[List[NDArray[np.float64]]] = None,
) -> float:
    """Computes the log-determinant of a positive definite matrix with eigenvalues smaller than one.

    Args:
        matrix (csr_array | NDArray): positive definite determinant
        n_sample (int): number of random samples
        n_degree (int): polynomial degree of chebyshev approximation
        delta (float): the eigenvalues of matrix have to be in the interval [delta, 1-delta]
        eigenvalues_deflate (Optional[List[float]]): The list of eigenvalues to remove
        eigenvectors_deflate (Optional[List[NDArray[np.float64]]]): The list of eigenvectors to remove


    Returns:
        float: the log determinant
    """

    eigenvalues_deflate_scaled = None
    n_deflate = 0
    if not eigenvalues_deflate is None:
        eigenvalues_deflate_scaled = [ev / sigma_max for ev in eigenvalues_deflate]
        n_deflate = len(eigenvalues_deflate)

    log_det = log_det_positive_definite_unit_interval(
        matrix=matrix / sigma_max,
        n_sample=n_sample,
        n_degree=n_degree,
        delta=0.0,
        eigenvalues_deflate=eigenvalues_deflate_scaled,
        eigenvectors_deflate=eigenvectors_deflate,
    )

    d = matrix.shape[0] - n_deflate

    return log_det + d * np.log(sigma_max)
