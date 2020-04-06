import numpy as np
from numpy import linalg as la
from Ffun import f
from propagator import propagate


def __direction_isolator(delta, i):
    m = np.zeros((6, 6))
    m[i][i] = 1
    return np.matmul(m, delta)


def derivative(x, delta, dt, obs_params, prop_params):
    n = 2
    m = 6

    a = np.zeros((n, m))
    for j in range(0, m):
        x1 = x + __direction_isolator(delta, j)
        temp1 = propagate(x + __direction_isolator(delta, j), dt, prop_params)
        temp2 = propagate(x - __direction_isolator(delta, j), dt, prop_params)
        temp3 = (f(temp1, obs_params) - f(temp2, obs_params)) / (2 * delta[j])

        for i in range(0, n):
            a[i][j] = temp3[i]

    return a


def milano(x, xoffset, dt, obs_params, prop_params):
    dr = .1
    dv = .001
    delta = np. array([dr, dr, dr, dv, dv, dv])

    yobs = f(propagate(x, dt, prop_params), obs_params)
    ypred = f(propagate(x+xoffset, dt, prop_params), obs_params)

    xi = yobs - ypred

    hello = np.zeros(20, 1)

    for i in range(0, 10):  # stopping criteria, should be changed to be based on deltaX
        hello[i] = la.norm(xi)

        b = -derivative(x, delta, dt, obs_params, prop_params)
        c = np.matmul(b.transpose(), b)
        d = -np.matmul(b.transpose(), xi)

        deltax = np.matmul(la.inv(c), d)

        xnew = x + deltax

        ypred = f(propagate(xnew, dt, prop_params), obs_params)
        x = xnew
        xi = yobs - ypred

    return xnew
