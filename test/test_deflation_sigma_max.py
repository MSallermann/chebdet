import numpy as np
from chebdet import chebdet, util
import scipy as sp


def test():
    n = int(1e2)

    print(f"Testing a {n} by {n} matrix")

    lambda_min = 1
    lambda_max = 10

    n_sample = 100
    n_degree = 100

    # define deflation vectors
    eigenvalues_deflate = [-10.0, 0.0]
    n_deflate = len(eigenvalues_deflate)
    eigenvectors_deflate = [util.standard_basis_vector(i, n) for i in range(n_deflate)]

    # Test with deflation
    diag = np.linspace(lambda_min, lambda_max, n)
    diag[:n_deflate] = eigenvalues_deflate

    B = sp.sparse.diags(diag)

    logdet = np.sum(np.log(diag[n_deflate:]))

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=B,
        n_sample=n_sample,
        n_degree=n_degree,
        sigma_max=lambda_max * 1.1,
        eigenvalues_deflate=eigenvalues_deflate,
        eigenvectors_deflate=eigenvectors_deflate,
    )

    print(f"{logdet = }")
    print(f"{logdet_algorithm = }")

    assert np.isclose(logdet, logdet_algorithm)
