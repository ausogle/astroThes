import numpy as np
from numpy import linalg as la
import scipy as sp
from src.ffun import f
from src.propagator import propagate
from src.dto import ObsParams, PropParams


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

    max_iter = 5
    delta_x = np.ones(len(x))               #Must break the stopping criteria
    hello = np.zeros((max_iter, 1))
    i = 0
    while not stopping_criteria(delta_x) and i < max_iter:
        hello[i] = la.norm(xi)

        b = -derivative(x, delta, obs_params, prop_params)
        c = b.transpose() @ b
        d = -b.transpose() @ xi
        # Built in inv() doesnt work
        # invC = la.inv(c)
        # deltax = invC @ d

        # Tried using np.linalg.svd() still doesnt work
        # invC = invert_svd(c)
        # deltax = invC @ d

        # Tried using np.linalg.qr() Also deosn't work
        # deltax = get_delta_x_from_qr_factorization(c, d)

        # Tried using gauss-seidel approach to solve system of equations C * delta_x = d
        # delta_x = get_delta_x_from_gauss_seidel(c, d)
        # xnew = x + delta_x

        # Tried inverting using ldl (scipy)
        invC = invert_using_ldl(c)
        delta_x = invC @ d
        xnew = x + delta_x

        # Tried using cholesky decomposition
        # invC = invert_using_cholesky(c)
        # delta_x = invC @ d
        # xnew = x + delta_x

        ypred = f(propagate(xnew, prop_params), obs_params)
        x = xnew - np.zeros(len(x))
        xi = yobs - ypred

        i = i+1
        if i == max_iter - 1:
            print("CAUTION: REACHED MAX ITERATIONS IN MILANI METHOD")

    print("The norm of the residuals has been", hello[0:i].transpose())
    print("\nThe main iteration sequence ran ", i, " times\n")
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
        x1 = x + direction_isolator(delta, j)
        temp1 = propagate(x + direction_isolator(delta, j), prop_params)
        temp2 = propagate(x - direction_isolator(delta, j), prop_params)
        temp3 = (f(temp1, obs_params) - f(temp2, obs_params)) / (2 * delta[j])

        for i in range(0, n):
            a[i][j] = temp3[i]

    return a


# def invert_using_cholesky(a: np.matrix) -> np.matrix:
#     l = np.linalg.cholesky(a)
#     invl = np.linalg.inv(l)
#     return invl.T@invl


def get_delta_x_from_qr_factorization(a: np.matrix, b: np.ndarray) -> np.ndarray:
    """
    Solves the linear equation C*deltax = d, comparable to Ax=b, using qr factorization from numpy.
    :param a: Is actually the c matrix from milani(), named to a for analogy to equation above.
    :param b: Is actually the d matrix from milani().
    :return: Returns delta_x or x from analogy
    """
    q, r = np.linalg.qr(a.transpose())
    x = q @ np.linalg.inv(r.transpose()) @ b
    return x


def invert_svd(a: np.matrix) -> np.matrix:
    """
    Inverts a matrix using singular value decomposition function in numpy
    :param a: matrix that needs to be inverted
    :return: multiplicative inverse of a
    """
    u, s, vh = np.linalg.svd(a)
    d = np.diag(s)
    ainv = vh.transpose() @ np.linalg.inv(d) @ u.transpose()
    return ainv


def get_delta_x_from_gauss_seidel(a: np.matrix, b: np.ndarray, max_iter=20, tol=1e-16) -> np.ndarray:
    """
    Gauss-Seidel approach to solve the linear system of equations Ax = b. In our specific case this is C*delta_x = d.
    Divergence is not uncommon from testing.
    :param a: c matrix from milani()
    :param b: d matrix from minlani()
    :param max_iter: maximum number of iterations acceptable
    :param tol: maximum tolerance for a single update for every element
    :return: returns delta_x
    """
    n = len(a)
    x = np.zeros(n)
    x_old = np.zeros(n)
    i = 0
    error = tol*2
    while i < max_iter and error > tol:
        for j in range(0, n):
            d = b[j]
            for k in range(0, n):
                if j != k:
                    d = d - a[j][k] * x[k]
            x[j] = d/a[j][j]
        delta_x = x - x_old
        error = max(abs(delta_x))
        x_old = x - np.zeros(n)
        i = i+1
    if i == max_iter - 1:
        print("CAUTION: REACHED MAX ITERATIONS IN GAUSS-SEIDEL METHOD")
    b_prime = a @ x
    print("Error observation in Gauss-Seidel Method\n", b.transpose(), "\n", b_prime.transpose())
    return x


def invert_using_ldl(a: np.matrix) -> np.matrix:
    lu, d, perm = sp.linalg.ldl(a, lower=0)
    invlu = np.linalg.inv(lu)
    invd = invert_diag(d)
    invlut = np.linalg.inv(lu.T)
    inva = invlut @ invd @ invlu
    return inva


def invert_diag(d: np.matrix) -> np.matrix:
    n = d.shape[1]
    for i in range(0, n):
        if d[i][i] > 1e-14:
            d[i][i] = 1 / d[i][i]
    return d


def stopping_criteria(delta_x: np.ndarray, rtol=1e-3, vtol=1e-6) -> bool:
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
