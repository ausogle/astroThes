import numpy as np
from scipy import linalg as la
from src.ffun import f
from src.propagator import propagate
from src.dto import ObsParams, PropParams
from scipy.linalg import solve_banded


def milani(x: np.ndarray, yobs: np.ndarray, obs_params: ObsParams, prop_params: PropParams, dr=.1, dv=.005) -> np.ndarray:
    """
    Scheme outlined in Adrea Milani's 1998 paper "Asteroid Idenitification Problem". It is a least-squared psuedo-newton
    approach to improving a objects's orbit description based on differences in object's measurement in the sky versus
    where it was predicted to be.
    :param x: State vector of the satellite at a time separate from the observation
    :param yobs: Observed values of the satellite, same format as prediction function Ffun()
    :param obs_params: Observational parameters, passed directly to Ffun()
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param dr: Spatial resolution to be used in derivative function.
    :param dv: Resolution used for velocity in derivative function
    :return: A more accurate state vector at the same time as original description, not observation
    """

    delta = np. array([dr, dr, dr, dv, dv, dv])

    ypred = f(propagate(x, prop_params), obs_params)

    xi = yobs - ypred

    max_iter = 15
    delta_x = np.ones(len(x))               #Must break the stopping criteria
    hello = []
    hello.append(la.norm(xi))
    i = 0
    while not stopping_criteria(delta_x) and i < max_iter:
        b = -derivative(x, delta, obs_params, prop_params)
        c = b.T @ b
        d = -b.T @ xi
        delta_x = get_delta_x(c, d)
        xnew = x + delta_x

        ypred = f(propagate(xnew, prop_params), obs_params)
        x = xnew - np.zeros(len(x))
        xi = yobs - ypred

        i = i+1
        hello.append(la.norm(xi))
        if i == max_iter - 1:
            print("CAUTION: REACHED MAX ITERATIONS IN MILANI METHOD")
    print("hello")
    print(hello)
    return xnew


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


def derivative(x: np.ndarray, delta: np.ndarray, obs_params: ObsParams, prop_params: PropParams, n=2) -> np.matrix:
    """
    Derivative() calculates derivatives of the prediction function per variable in the state vector and returns a matrix
    where each element is the column corresponds to an element of the prediction function output and the row corresponds
    to an element being varied in the state vector. Uses a second-order centered difference equation.
    :param x: State vector
    :param delta: variation in position/velocity to be used in derivatives
    :param obs_params: Observational parameters, passed directly to Ffun0(
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param n: number of elements in prediction function output
    :return: Matrix of derivatives of the prediction function in position/velocity space
    """
    m = len(x)

    a = np.zeros((n, m))
    for j in range(0, m):
        temp1 = propagate(x + direction_isolator(delta, j), prop_params)
        temp2 = propagate(x - direction_isolator(delta, j), prop_params)
        temp3 = (f(temp1, obs_params) - f(temp2, obs_params)) / (2 * delta[j])

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
    residual = a@x-b
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


def stopping_criteria(delta_x: np.ndarray, rtol=1e-2, vtol=1e-5) -> bool:
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
