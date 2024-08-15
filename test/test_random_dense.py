from chebdet import chebdet
import numpy as np
import scipy as sp
from numpy.typing import NDArray
import util

def test():
    n = 20

    print(f"Testing a {n} by {n} matrix")

    lambda_min = 0.1
    lambda_max = 0.9
    delta = 0.01

    n_sample = 60
    n_degree = 40

    diag = np.linspace(lambda_min, lambda_max, n)

    # B = util.generate_matrix(spectrum=diag)
    B = sp.sparse.diags(diag)

    logdet = np.sum(np.log(diag))

    logdet_algorithm = chebdet.logdet(matrix=B, n_sample=n_sample, n_degree=n_degree, delta=delta)

    assert np.isclose(logdet, logdet_algorithm)

    print(f"{logdet = }")
    print(f"{logdet_algorithm = }")