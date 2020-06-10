import numpy as np
from astropy.time import Time
from src.dto import PropParams
from src.state_propagator import state_propagate
from src.core import direction_isolator


def cov_propagate(x: np.ndarray, epoch_t: Time, prop_params: PropParams, p_i: np.ndarray) -> np.ndarray:
    """
    This function propagates a covariance matrix through time.

    :param x: state vector covariance matrix is tied to at original epoch
    :param epoch_t: Final epoch
    :param prop_params: Parameters relevant to propagation. Super accurate propagation is not always required. Some
    textbooks suggest using variational equations to save time.
    :param p_i: Covariance matrix at initial epoch
    """
    dr = .1
    dv = .005
    delta = np.array([dr, dr, dr, dv, dv, dv])

    a = dx_dx0(x, epoch_t, prop_params, delta)
    p_t = a @ p_i @ a.T
    return p_t


def dx_dx0(x: np.ndarray, epoch_t: Time, prop_params: PropParams, delta: np.ndarray) -> np.ndarray:
    """
    Calculates partial derivatives of a state vector in the future based on a current state vector.

    :param x: state vector at original epoch
    :param epoch_t: Final epoch
    :param prop_params: Parameters relevant to propagation
    :param delta: Array holding step sizes for derivatives
    """
    n = len(x)
    a = np.zeros((n, n))

    for j in range(0, n):
        temp1 = state_propagate(x + direction_isolator(delta, j), epoch_t, prop_params)
        temp2 = state_propagate(x - direction_isolator(delta, j), epoch_t, prop_params)
        temp3 = temp1 - temp2

        for i in range(0, n):
            a[i][j] = temp3[i] / (2 * delta[i])
    return a
