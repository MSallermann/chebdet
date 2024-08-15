import numpy as np
from numpy.typing import NDArray
from typing import List


def standard_basis_vector(i: int, n: int) -> NDArray[np.float64]:
    """Get a vector of the cartesian standard basis

    Args:
        i (int): _description_
        n (int): _description_

    Returns:
        NDArray[np.float64]: _description_
    """
    res = np.zeros((n, 1), dtype=np.float64)
    res[i, 0] = 1.0
    return res


def generate_matrix(spectrum: List[float]) -> NDArray[np.float64]:
    """
    Generates a matrix with a specific spectrum by applying the
    qr algoirhtm to a random matrix and rotating a diagonal matrix
    by the resulting basis
    """
    diagonalized = np.diag(spectrum)
    basis = np.random.rand(len(spectrum), len(spectrum))
    basis = np.linalg.qr(basis)[0]
    return basis.T @ diagonalized @ basis
