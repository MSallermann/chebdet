from chebdet import chebdet
import numpy as np
import scipy as sp
from numpy.typing import NDArray
import util


if __name__ == "__main__":
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

    Gamma = cheb_det.logdet(matrix=B, n_sample=n_sample, n_degree=n_degree, delta=delta)

    print("From algorithm: ", Gamma)
    print("Theoretical det: ", logdet)

    # define deflation vectors
    eigenvalues_deflate = [-1.0, 0.0]
    n_deflate = len(eigenvalues_deflate)
    eigenvectors_deflate = [util.standard_basis_vector(i, n) for i in range(n_deflate)]

    # Test with deflation
    diag = np.linspace(lambda_min, lambda_max, n)
    diag[:n_deflate] = eigenvalues_deflate

    B = sp.sparse.diags(diag)

    logdet = np.sum(np.log(diag[n_deflate:]))

    Gamma = cheb_det.logdet(
        matrix=B,
        n_sample=n_sample,
        n_degree=n_degree,
        delta=delta,
        eigenvalues_deflate=eigenvalues_deflate,
        eigenvectors_deflate=eigenvectors_deflate,
    )

    print("From algorithm: ", Gamma)
    print("Theoretical det: ", logdet)

    # import matplotlib.pyplot as plt

    # delta_list = np.linspace(0.0, 0.1, 10)

    # m, n = 10, 10
    # delta = 0.1
    # m_list, n_list = range(1, 30, 1), range(5, 20, 1)

    # fig, ax = plt.subplots(1, 3, sharey=True)
    # fig.set_size_inches(18, 6)
    # Gamma_List = [cheb_det.logdet(B, m, n, d) for d in delta_list]
    # ax[0].plot(delta_list, Gamma_List, label="m={}, n={}".format(m, n), marker=".")
    # ax[0].axhline(
    #     logdet,
    #     ls="--",
    #     color="black",
    #     label="log det B = {:.2f}".format(logdet),
    # )
    # ax[0].set_xlabel(r"$\delta$")
    # ax[0].set_ylabel(r"$\Gamma$")
    # ax[0].legend()

    # Gamma_List = [cheb_det.logdet(B, mc, n, delta) for mc in m_list]
    # ax[1].plot(
    #     m_list, Gamma_List, label=r"n={}, $\delta$={}".format(n, delta), marker="."
    # )
    # ax[1].axhline(
    #     logdet,
    #     ls="--",
    #     color="black",
    #     label="log det B = {:.2f}".format(logdet),
    # )
    # ax[1].set_xlabel("m")
    # ax[1].set_ylabel(r"$\Gamma$")
    # ax[1].legend()

    # Gamma_List = [cheb_det.logdet(B, m, n, delta) for n in n_list]
    # ax[2].plot(
    #     n_list, Gamma_List, label=r"m={}, $\delta$={}".format(m, delta), marker="."
    # )
    # ax[2].axhline(
    #     logdet,
    #     ls="--",
    #     color="black",
    #     label="log det B = {:.2f}".format(logdet),
    # )
    # ax[2].set_xlabel("n")
    # ax[1].set_ylabel(r"$\Gamma$")
    # ax[2].legend()

    # plt.savefig("plot", dpi=300)
    # plt.show()
