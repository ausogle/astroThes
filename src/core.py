import numpy as np
from scipy import linalg as la
from src.ffun import f
from src.propagator import propagate
from src.dto import ObsParams, PropParams


def milani(x: np.ndarray, xoffset: np.ndarray, obs_params: ObsParams, prop_params: PropParams, dr=.1, dv=.005) -> np.ndarray:
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

    max_iter = 3
    delta_x = np.ones(len(x))               #Must break the stopping criteria
    hello = np.zeros((max_iter+1, 1))
    hello[0] = la.norm(xi)
    i = 0
    while not stopping_criteria(delta_x) and i < max_iter:
        b = -derivative(x, delta, obs_params, prop_params)
        c = b.T @ b
        d = -b.T @ xi

        # inv_c = invert_using_lu(c)
        # delta_x = inv_c @ d
        delta_x = get_delta_x(c, d)
        xnew = x + delta_x

        ypred = f(propagate(xnew, prop_params), obs_params)
        x = xnew - np.zeros(len(x))
        xi = yobs - ypred

        i = i+1
        hello[i] = la.norm(xi)
        if i == max_iter - 1:
            print("CAUTION: REACHED MAX ITERATIONS IN MILANI METHOD")
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
        temp4 = f(temp1, obs_params)
        temp5 = f(temp2, obs_params)
        temp3 = (f(temp1, obs_params) - f(temp2, obs_params)) / (2 * delta[j])

        for i in range(0, n):
            a[i][j] = temp3[i]

    return a


def invert_using_lu(a: np.matrix) -> np.matrix:
    """
    Inverts a matrix using the numpy.linalg.lu factorization.
    :param a: Matrix desired to be inverted. Will be the c matrix.
    """
    p, l, u = la.lu(a)
    invu = la.inv(u)
    return la.inv(u) @ la.inv(l) @ p.T


def get_delta_x(a,b):
    p, l, u = la.lu(a)
    invl = la.inv(l)
    x_triag = la.solve_triangular(u, invl @ p.T @ b)
    residual_triag = a @ x_triag - b
    print(residual_triag)

    # x_orig = la.inv(u) @ la.inv(l) @ p.T @ b
    # residual_orig = a@x_orig-b
    return x_triag


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
