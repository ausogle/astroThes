import numpy as np
from numpy import linalg as la
from src.ffun import f
from src.propagator import propagate
from src.dto import ObsParams, PropParams


def __direction_isolator(delta: np.ndarray, i: int):
    """
    __direction_isolator() manipulates the delta array to return an empty array with the exc
    :param delta: Input array of step sizes for calculating derivatives around the state vector
    :param i: the element of delta desired to be preserved.
    :return: Near empty array, where the ith element is the ith element of delta.
    """
    m = np.zeros((6, 6))
    m[i][i] = 1
    return np.matmul(m, delta)


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
        x1 = x + __direction_isolator(delta, j)
        temp1 = propagate(x + __direction_isolator(delta, j), prop_params)
        temp2 = propagate(x - __direction_isolator(delta, j), prop_params)
        temp3 = (f(temp1, obs_params) - f(temp2, obs_params)) / (2 * delta[j])

        for i in range(0, n):
            a[i][j] = temp3[i]

    return a


def get_delta_x_from_qr_factorization(a: np.matrix, b: np.ndarray) -> np.ndarray:
    """
    Solves the linear equation C*deltax = d, comparable to Ax=b, using qr factorization from numpy.
    :param a: Is actually the c matrix from milani(), named to a for analogy to equation above.
    :param b: Is actually the d matrix from milani().
    :return: Returns delta_x or x from analogy
    """
    q, r = np.linalg.qr(a.transpose())
    x = np.matmul(q, np.matmul(np.linalg.inv(r.transpose()), b))
    return x


def invert_svd(a: np.matrix) -> np.matrix:
    u, s, vh = np.linalg.svd(a)
    d = np.diag(s)
    ainv = np.matmul(np.matmul(vh.transpose(), np.linalg.inv(d)), u.transpose())
    return ainv


def milani(x: np.ndarray, xoffset: np.ndarray, obs_params: ObsParams, prop_params: PropParams, dr=.1, dv=.001) -> np.ndarray:
    """
    Scheme outlined in Adrea Milani's 1998 paper "Asteroid Idenitification Problem". It is a least-squared psuedo-newton
    approach to improving a objects's orbit description based on differences in object's measurement in the sky versus
    where it was predicted to be.
    :param x: State vector of the satellite at a time separate from the observation
    :param xoffset: observational offset. Will be changed eventually to the observation values. Same format as output
    of the prediction function
    :param obs_params: Observational parameters, passed directly to Ffun()
    :param prop_params: Propagation parameters, passed directly to propagate()
    :param dr: Spatial resolution to be used in derivative function.
    :param dv: Resolution used for velocity in derivative function
    :return: A more accurate state vector at the same time as original description, not observation
    """

    delta = np. array([dr, dr, dr, dv, dv, dv])

    yobs = f(propagate(x, prop_params), obs_params)
    ypred = f(propagate(x+xoffset, prop_params), obs_params)

    xi = yobs - ypred

    hello = np.zeros((20, 1))

    for i in range(0, 3):  # stopping criteria, should be changed to be based on deltaX
        hello[i] = la.norm(xi)

        b = -derivative(x, delta, obs_params, prop_params)
        c = np.matmul(b.transpose(), b)
        d = -np.matmul(b.transpose(), xi)

        # invC = la.inv(c)
        # deltax = np.matmul(invC, d)
        invC = invert_svd(c)
        deltax = np.matmul(invC, d)
        # deltax = get_delta_x_from_qr_factorization(c, d)
        xnew = x + deltax

        ypred = f(propagate(xnew, prop_params), obs_params)
        x = xnew
        xi = yobs - ypred

    print(hello)
    return xnew
