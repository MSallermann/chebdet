import numpy as np
import util
from chebdet import chebdet
import scipy as sp

def test():
    n = int(1e4)

    print(f"Testing a {n} by {n} matrix")

    lambda_min = 0.1
    lambda_max = 0.9
    delta = 0.01

    n_sample = 60
    n_degree = 40

    # define deflation vectors
    eigenvalues_deflate = [-1.0, 0.0]
    n_deflate = len(eigenvalues_deflate)
    eigenvectors_deflate = [util.standard_basis_vector(i, n) for i in range(n_deflate)]

    # Test with deflation
    diag = np.linspace(lambda_min, lambda_max, n)
    diag[:n_deflate] = eigenvalues_deflate

    B = sp.sparse.diags(diag)

    logdet = np.sum(np.log(diag[n_deflate:]))

    logdet_algorithm = chebdet.logdet(
        matrix=B,
        n_sample=n_sample,
        n_degree=n_degree,
        delta=delta,
        eigenvalues_deflate=eigenvalues_deflate,
        eigenvectors_deflate=eigenvectors_deflate,
    )

    # assert False

    assert np.isclose(logdet, logdet_algorithm)

    print(f"{logdet = }")
    print(f"{logdet_algorithm = }")
