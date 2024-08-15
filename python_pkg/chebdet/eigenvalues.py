from typing import Union
import scipy as sp
from scipy.sparse import csr_array
from numpy.typing import NDArray
import numpy as np


def max_eval_norm(matrix: Union[csr_array, NDArray]):
    res = sp.sparse.linalg.norm(matrix, ord="fro")
    return res


def sigma_max(matrix: Union[csr_array, NDArray], k=1):
    vals, vecs = sp.sparse.linalg.eigs(
        matrix,
        k=k,
        M=None,
        sigma=None,
        which="LR",
        v0=None,
        ncv=None,
        maxiter=None,
        tol=0,
        return_eigenvectors=True,
        Minv=None,
        OPinv=None,
        OPpart=None,
    )
    return np.real(vals), np.real(vecs)


def sigma_min(matrix: Union[csr_array, NDArray], k=1):
    vals, vecs = sp.sparse.linalg.eigs(
        matrix,
        k=k,
        M=None,
        sigma=None,
        which="SR",
        v0=None,
        ncv=None,
        maxiter=None,
        tol=0,
        return_eigenvectors=True,
        Minv=None,
        OPinv=None,
        OPpart=None,
    )
    return np.real(vals), np.real(vecs)
