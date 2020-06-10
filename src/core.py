import numpy as np
from scipy import linalg as la
from src.observation_function import y
from src.state_propagator import state_propagate
from src.dto import Observation, PropParams
from scipy.linalg import solve_banded
from typing import Tuple, List


def milani(x: np.ndarray, observations: List[Observation], prop_params: PropParams,
           l=np.zeros((6, 6)), dr=.1, dv=.005, max_iter=15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scheme outlined in Adrea Milani's 1998 paper "Asteroid Idenitification Problem". It is a least-squared psuedo-newton
    approach to improving a objects's orbit description based on differences in object's measurement in the sky versus
    where it was predicted to be.

    :param x: State vector of the satellite at a time separate from the observation
    :param observations: List of observational objects that capture location, time, and direct observational parameters.
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param l: Information matrix. Inverse of covariance matrix. Represents uncertainty in initial state. Default value
    of zero matrix will leave the code unaffected. l_kk ~ 1/sigma_k^s.
    :param dr: Spatial resolution to be used in derivative function.
    :param dv: Resolution used for velocity in derivative function
    :param max_iter: Maximum number of iterations for Least Squares filter
    :return: A more accurate state vector at the same time as original description, not observation
    """

    n = len(x)
    delta = np. array([dr, dr, dr, dv, dv, dv])
    delta_x = np.ones(n)  # Must break the stopping criteria

    i = 0
    while not stopping_criteria(delta_x) and i < max_iter:
        c = np.zeros((n, n))
        d = np.zeros((n, 1))
        for observation in observations:
            ypred = y(state_propagate(x, observation.epoch, prop_params), observation)
            yobs = observation.obs_values
            xi = yobs - ypred
            w = np.diag(1/np.multiply(observation.obs_sigmas, observation.obs_sigmas))

            b = -dy_dstate(x, delta, observation, prop_params)
            c += b.T @ w @ b
            d += -b.T @ w @ xi

        delta_x = get_delta_x(l + c, d)
        xnew = x + delta_x
        x = xnew - np.zeros(n)
        i = i+1

    p = np.linalg.inv(l + c)
    # covariance_residual = p @ (l + c) - np.eye(n)

    return xnew, p


def direction_isolator(delta: np.ndarray, i: int):
    """
    direction_isolator() manipulates the delta array to return an empty array with the exc

    :param delta: Input array of step sizes for calculating derivatives around the state vector
    :param i: the element of delta desired to be preserved.
    :return: Near empty array, where the ith element is the ith element of delta.
    """
    m = np.zeros((6, 6))
    m[i][i] = 1
    return m @ delta


def dy_dstate(x: np.ndarray, delta: np.ndarray, observation: Observation, prop_params: PropParams, n=2) -> np.ndarray:
    """
    dy_dstate() calculates derivatives of the prediction function per state vector and returns a matrix where each
    element is the column corresponds to an element of the prediction function output and the row corresponds
    to an element being varied in the state vector. Uses a second-order centered difference equation.

    :param x: State vector
    :param delta: variation in position/velocity to be used in derivatives
    :param observation: Observational parameters, passed directly to Ffun0(
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param n: number of elements in prediction function output
    :return: Matrix of derivatives of the prediction function in position/velocity space
    """
    m = len(x)

    a = np.zeros((n, m))
    for j in range(0, m):
        temp1 = state_propagate(x + direction_isolator(delta, j), observation.epoch, prop_params)
        temp2 = state_propagate(x - direction_isolator(delta, j), observation.epoch, prop_params)
        temp3 = (y(temp1, observation) - y(temp2, observation)) / (2 * delta[j])

        for i in range(0, n):
            a[i][j] = temp3[i]
    return a


def get_delta_x(a: np.matrix, b: np.ndarray) -> np.ndarray:
    """
    Solves the system of equation using the scipy wrapper for LAPACK's dgbsv function.
    Requires converting a into ab matrix. Notably, for our system the upper and lower bandwidths are both 5.

    :param a: A matrix in normal equation. For our problem this is C
    :param b: b vector in normal equation. For our problem this is D
    """
    upper = 5
    lower = 5
    ab = diagonal_form(a, upper=upper, lower=lower)
    x = solve_banded((upper, lower), ab, b)
    # residual = a@x-b
    return x


def diagonal_form(a: np.matrix, upper=1, lower=1) -> np.matrix:
    """
    Ripped from github.com/scipy/scipy/issues/8362. User Khalilsqu wrote the following function.
    Converts a into ab given upper and lower bandwidths.
    Follows notes at people.sc.kuleuven.be/~raf.vanderbril/homepage/publications/papers_html/fr_lev/node16.html

    :param a: A matrix in normal equation. For our problem this is C
    :param upper: Upper bandwidth of a
    :param lower: Lower bandwidth of a
    """
    n = a.shape[1]
    ab = np.zeros((2*n-1, n))
    for i in range(n):
        ab[i, (n-1)-i:] = np.diagonal(a, (n-1)-i)

    for i in range(n-1):
        ab[(2*n-2)-i, :i+1] = np.diagonal(a, i-(n-1))
    mid_row_inx = int(ab.shape[0]/2)
    upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
    keep_rows = upper_rows + lower_rows
    ab = ab[keep_rows, :]
    return ab


def stopping_criteria(delta_x: np.ndarray, rtol=1e-6, vtol=1e-9) -> bool:
    """
    Determines whether or not the algorithm can stop. Currently evaluates against arbitrary conditions. To fully
    integrate rtol and vtol into code, they need to be included in one of the params objects. Tests if the position and
    velocity are within a certain distance of the previous iteration. Assuming with each step we get closer, this
    implies we were within the specified tolerances supplied, or assumed above.

    :param delta_x: difference in state vector from previous interation
    :param rtol: tolerance on change in position [km]
    :param vtol: tolerance on change in velocity [km/s]
    :return: returns if change is within tolerance
    """
    r = np.linalg.norm(delta_x[0:3])
    v = np.linalg.norm(delta_x[3:6])
    return r < rtol and v < vtol
