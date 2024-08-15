import numpy as np
from chebdet import chebdet, eigenvalues, util
import scipy as sp
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
    n_degree = 10

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=hessian,
        n_sample=n_sample,
        n_degree=n_degree,
        sigma_max=sigma_max[0],
        eigenvalues_deflate=[sigma_min],
        eigenvectors_deflate=[vec_min.reshape((n, 1))],
    )

    logdet_lu = util.get_logdet(hessian) - np.log(np.abs(sigma_min[0]))
    rel_err = np.abs((logdet_algorithm - logdet_lu) / logdet_lu)

    print(f"{logdet_lu = }")
    print(f"{logdet_algorithm = }")
    print(f"{rel_err = :.1e}")

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
    n_degree = 10

    logdet_algorithm = chebdet.log_det_positive_definite(
        matrix=hessian,
        n_sample=n_sample,
        n_degree=n_degree,
        sigma_max=sigma_max[0],
        eigenvalues_deflate=sigma_min,
        eigenvectors_deflate=[vm.reshape((n, 1)) for vm in vec_min.T],
    )

    logdet_lu = util.get_logdet(hessian) - sum(np.log(sigma_min))
    rel_err = np.abs((logdet_algorithm - logdet_lu) / logdet_lu)

    print(f"{logdet_lu = }")
    print(f"{logdet_algorithm = }")
    print(f"{rel_err = :.1e}")

    assert np.isclose(logdet_lu, logdet_algorithm, rtol=1e-3)
