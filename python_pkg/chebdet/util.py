import numpy as np
from numpy.typing import NDArray
from typing import List
from pathlib import Path
from scipy.sparse import coo_array, csc_array
from scipy.sparse.linalg import splu


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
    QR algoirthm to a random matrix and rotating a diagonal matrix
    by the resulting basis
    """
    diagonalized = np.diag(spectrum)
    basis = np.random.rand(len(spectrum), len(spectrum))
    basis = np.linalg.qr(basis)[0]
    return basis.T @ diagonalized @ basis


def read_sparse_matrix(path_to_matrix: Path, n_rows: int, n_cols: int) -> coo_array:
    """Read a sparse matrix from a file of triplets"""

    triplets = np.loadtxt(path_to_matrix)
    rows = np.round(triplets[:, 0], decimals=0)
    cols = np.round(triplets[:, 1], decimals=0)
    data = triplets[:, 2]

    matrix = coo_array((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float64)
    return matrix


def get_logdet(M: csc_array) -> float:
    """Computes the log determinant of the sparse matrix M using a sparse LU decomposition"""
    lu = splu(M.tocsc())
    log_det = np.sum(np.log(np.abs(lu.L.diagonal())) + np.log(np.abs(lu.U.diagonal())))
    return log_det
