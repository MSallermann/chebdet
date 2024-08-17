from chebdet import chebdet, util, error_bounds
import numpy as np
import scipy as sp
from numpy.typing import NDArray


def test():
    n = 20

    print(f"Testing a {n} by {n} matrix")

    lambda_min = 0.1
    lambda_max = 0.9
    delta = 0.1

    n_sample = int(1e4)
    n_degree = 20

    diag = np.linspace(lambda_min, lambda_max, n)

    B = util.generate_matrix(spectrum=diag)

    logdet = np.sum(np.log(diag))

    logdet_algorithm = chebdet.log_det_positive_definite_unit_interval(
        matrix=B, n_sample=n_sample, n_degree=n_degree, delta=delta
    )

    # Absolute error on logdet
    abs_log_error = np.abs(logdet - logdet_algorithm)

    # Relative error on the determinant (not the log)
    rel_error = np.exp(abs_log_error) - 1.0

    print(f"{logdet = }")
    print(f"{logdet_algorithm = }")

    print(f"{abs_log_error = :.1e}")
    print(f"{rel_error = :.1e}")

    assert np.isclose(rel_error, 0.0, atol=5e-2)
