import numpy as np


def min_parameters(epsilon: float, delta: float, gamma: float = 0.05):
    """Error bounds, according to Theorem 1 of Han et. al.

       Given epsilon, gamma in the interval (0,1), choosing
       n_degree > n_degree_min and n_sampling > n_sampling_min
       guarantees the following:

       Pr( | logdet - G | < epsilon logdet  ) > 1 - gamma

       where G is the result of chebdet.log_det_positive_definite_unit_interval.

       NOTE: The equation for n_degree_min is tight, while the equation for n_sampling is not and much lower sampling counts could be needed.

    Args:
        epsilon (float): the required relative error in the log determinant
        delta (float): the eigenvalue bounds lambda in [delta, 1-delta]
        gamma (float): the probability for the result to be worse than the requested relative error epsilon

    Returns:
        Tuple[float,float]: n_sampling_min, n_degree_min
    """

    n_sampling_min = int(54.0 / epsilon**2 * np.log(2.0 / gamma))
    n_degree_min = int(np.sqrt(1.0 / delta * np.log(1.0 / (epsilon * delta))))

    return n_sampling_min, n_degree_min


def convert_relative_errors(epsilon_det: float, logdet: float) -> float:
    """Convert a relative error on the determinant to a corresponding relative error on the log determinant"""
    epsilon_logdet = np.log(1.0 + epsilon_det) / logdet
    return epsilon_logdet
