from chebdet import chebdet, util
import numpy as np
import scipy as sp
from numpy.typing import NDArray


def test():
    n = 20

    print(f"Testing a {n} by {n} matrix")

    lambda_min = 1
    lambda_max = 9
    # delta = 0.01

    n_sample = 120
    n_degree = 80

    diag = np.linspace(lambda_min, lambda_max, n)

    # B = util.generate_matrix(spectrum=diag)
    B = sp.sparse.diags(diag)

    logdet = np.sum(np.log(diag))

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=B, n_sample=n_sample, n_degree=n_degree, sigma_max=lambda_max * 1.1
    )

    print(f"{logdet = }")
    print(f"{logdet_algorithm = }")

    assert np.isclose(logdet, logdet_algorithm)
