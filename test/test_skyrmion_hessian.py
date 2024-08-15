import numpy as np
import util
from chebdet import chebdet, eigenvalues
import scipy as sp
import util
from pathlib import Path


def test_sp():
    n = 2 * 100**2

    res_folder = Path(__file__).parent / "res"

    hessian = util.read_sparse_matrix(res_folder / "hessian_2n_sp", n, n).tocsr()

    print(f"Testing a {n} by {n} matrix")

    sigma_max = sp.sparse.linalg.norm(hessian, ord="fro")
    print(f"estimated sigma_max = {sigma_max}")

    sigma_min, vec_min = eigenvalues.sigma_min(hessian)
    sigma_max, vec_max = eigenvalues.sigma_max(hessian)

    print(f"{sigma_min = }")
    print(f"{sigma_max = }")

    n_sample = 1000
    n_degree = 40

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=hessian,
        n_sample=n_sample,
        n_degree=n_degree,
        sigma_max=sigma_max,
        eigenvalues_deflate=[sigma_min],
        eigenvectors_deflate=[vec_min.reshape((n, 1))],
    )

    logdet_lu = util.get_logdet(hessian) - np.log(np.abs(sigma_min))

    print(f"{logdet_lu = }")
    print(f"{logdet_algorithm = }")

    assert np.isclose(logdet_lu, logdet_algorithm, rtol=1e-3)


def test_min():
    n = 2 * 100**2

    res_folder = Path(__file__).parent / "res"

    hessian = util.read_sparse_matrix(res_folder / "hessian_2n_min", n, n).tocsr()

    print(f"Testing a {n} by {n} matrix")

    sigma_max = sp.sparse.linalg.norm(hessian, ord="fro")
    print(f"estimated sigma_max = {sigma_max}")

    sigma_min, vec_min = eigenvalues.sigma_min(hessian, k=2)
    sigma_max, vec_max = eigenvalues.sigma_max(hessian)

    print(f"{sigma_min = }")
    print(f"{sigma_max = }")

    n_sample = 1000
    n_degree = 40

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=hessian,
        n_sample=n_sample,
        n_degree=n_degree,
        sigma_max=sigma_max,
        eigenvalues_deflate=sigma_min,
        eigenvectors_deflate=[vm.reshape((n, 1)) for vm in vec_min.T],
    )

    print(sum(np.log(sigma_min)))

    logdet_lu = util.get_logdet(hessian) - sum(np.log(sigma_min))

    print(f"{logdet_lu = }")
    print(f"{logdet_algorithm = }")

    assert np.isclose(logdet_lu, logdet_algorithm, rtol=1e-3)
