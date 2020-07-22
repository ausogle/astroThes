import numpy as np
from src.observation_function import y
from src.state_propagator import state_propagate
from src.dto import Observation, PropParams, FilterOutput
from scipy.linalg import solve_banded
from typing import List


def milani(x: np.ndarray, observations: List[Observation], prop_params: PropParams,
           a_priori=FilterOutput(), dr=1, dv=.05, max_iter=15) -> FilterOutput:
    """
    Scheme outlined in Adrea Milani's 1998 paper "Asteroid Idenitification Problem". It is a least-squared psuedo-newton
    approach to improving a objects's orbit description based on differences in object's measurement in the sky versus
    where it was predicted to be.

    :param x: State vector of the satellite at a time separate from the observation
    :param observations: List of observational objects that capture location, time, and direct observational parameters.
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param a_priori: Output from a previous iteration
    :param dr: Spatial resolution to be used in derivative function
    :param dv: Resolution used for velocity in derivative function
    :param max_iter: Maximum number of iterations for Least Squares filter
    :return: A more accurate state vector at the same time as original description, not observation
    """

    n = len(x)
    x_in = x - np.zeros(6)
    delta = np. array([dr, dr, dr, dv, dv, dv])
    delta_x = np.ones(n)
    rms_old = 1e10                  #Following two definitions must break loop for first two iterations
    rms_new = 1e8

    i = 0
    while not stopping_criteria(rms_new, rms_old) and i < max_iter:
        c = np.zeros((n, n))
        d = np.zeros(n)
        rms_old = rms_new - 0
        for observation in observations:
            ypred = y(state_propagate(x, observation.epoch, prop_params), observation)
            yobs = observation.obs_values
            xi = yobs - ypred
            w = np.diag(1/np.multiply(observation.obs_sigmas, observation.obs_sigmas))

            b = dy_dstate(x, delta, observation, prop_params)
            c += b.T @ w @ b
            d += b.T @ w @ xi
        if np.array_equal(a_priori.p, np.zeros((6, 6))):
            delta_x = get_delta_x(c, d)
        else:
            l = get_inverse(a_priori.p)
            delta_x = get_delta_x(l + c, l @ a_priori.delta_x + d)
        xnew = x + delta_x
        x = xnew - np.zeros(n)
        rms_new = np.sqrt((xi.T @ w @ xi)/n)
        i = i+1

    p = a_priori.p + get_inverse(c)
    # covariance_residual = la.norm(p @ c - np.eye(6))
    # print("Covariance Residual")
    # print(covariance_residual)

    output = FilterOutput(x_in, prop_params.epoch, x, delta_x, p)
    return output


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


def get_delta_x(a: np.matrix, b: np.ndarray, upper=5, lower=5) -> np.ndarray:
    """
    Solves the system of equation using the scipy wrapper for LAPACK's dgbsv function.
    Requires converting a into ab matrix. Notably, for our system the upper and lower bandwidths are both 5.

    :param a: A matrix in normal equation. For our problem this is C
    :param b: b vector in normal equation. For our problem this is
    :param upper: Upper bandwidth of matric C
    :param lower: Lower bandwidth of matrix c
    """
    ab = diagonal_form(a, upper=upper, lower=lower)
    x = solve_banded((upper, lower), ab, b)
    residual = a@x-b
    print("residual")
    print(np.linalg.norm(residual))
    return x


def get_inverse(c: np.ndarray):
    """
    Inverts C matrix using banded solver.

    :param c: Normal matrix required to invert for covariance matrix
    """
    inv = get_delta_x(c, np.eye(6))
    # residual = c @ inv - np.eye(6)
    # print(np.linalg.norm(residual))
    return inv


def diagonal_form(a: np.matrix, upper=5, lower=5) -> np.matrix:
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


def stopping_criteria(rms_new: float, rms_old: float, tol=1e-1) -> bool:
    """
    Determines whether or not the algorithm can stop. Currently evaluates against arbitrary conditions. To fully
    integrate rtol and vtol into code, they need to be included in one of the params objects. Tests if the position and
    velocity are within a certain distance of the previous iteration. Assuming with each step we get closer, this
    implies we were within the specified tolerances supplied, or assumed above.

    :param rms_new: New root mean square from current iteration
    :param rms_old: Root mean square from previous iteration
    :param tol: relative tolerance between updates
    :return: returns if change is within relative tolerance
    """
    if rms_new == 0:
        return True
    percent_diff = np.abs((rms_old - rms_new)/rms_old)
    return percent_diff < tol
