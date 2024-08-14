import cheb_det
import numpy as np
import random

if __name__ == "__main__":
    n = 8
    B = np.eye(n)
    det = 1
    for i in range(n):
        B[i][i] = random.randint(1,9)/10
        det = det*B[i][i]

    Gamma = cheb_det.logdet(B, 30, 20, 0.08)
    print(B)
    print("From algorithm: ",  np.exp(Gamma))
    print("Theoretical det: ", det)#

    import matplotlib.pyplot as plt
    delta_list = np.linspace(0.0, 0.1, 10)

    m, n = 10, 10
    delta = 0.1
    m_list, n_list = range(1,30,1), range(5,20,1)

    fig, ax = plt.subplots(1,3,sharey=True)
    fig.set_size_inches(18,6)
    Gamma_List = [cheb_det.logdet(B, m, n, d) for d in delta_list]
    ax[0].plot(delta_list, Gamma_List, label = "m={}, n={}".format(m,n), marker=".")
    ax[0].axhline(np.log(det), ls="--", color="black", label= "log det B = {:.2f}".format(np.log(det)))
    ax[0].set_xlabel(r"$\delta$")
    ax[0].set_ylabel(r"$\Gamma$")
    ax[0].legend()

    Gamma_List = [cheb_det.logdet(B, mc, n, delta) for mc in m_list]
    ax[1].plot(m_list, Gamma_List, label = r"n={}, $\delta$={}".format(n,delta), marker=".")
    ax[1].axhline(np.log(det), ls="--", color="black", label= "log det B = {:.2f}".format(np.log(det)))
    ax[1].set_xlabel("m")
    ax[1].set_ylabel(r"$\Gamma$")
    ax[1].legend()

    Gamma_List = [cheb_det.logdet(B, m, n, delta) for n in n_list]
    ax[2].plot(n_list, Gamma_List, label = r"m={}, $\delta$={}".format(m,delta), marker=".")
    ax[2].axhline(np.log(det), ls="--",color="black", label= "log det B = {:.2f}".format(np.log(det)))
    ax[2].set_xlabel("n")
    ax[1].set_ylabel(r"$\Gamma$")
    ax[2].legend()

    plt.savefig("plot", dpi=300)
    plt.show()
