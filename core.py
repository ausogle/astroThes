import numpy as np
from numpy import linalg as la
from Ffun import f
from propagator import propagate


def __direction_isolator(delta, i):
    m = np.zeros(6)
    m[i][i] = 1
    return np.matmul(m, delta)


def __derivative(x, dt, params):
    n = 2
    m = 6
    dr = .1
    dv = .001
    delta = np. array([dr, dr, dr, dv, dv, dv])

    a = np.zeros(n, m)
    for j in range(0, m-1):
        temp1 = propagate(x + __direction_isolator(delta, j), dt, params)
        temp2 = propagate(x - __direction_isolator(delta, j), dt, params)
        temp3 = (f(temp1) - f(temp2))/(2*delta[j])

        for i in range(0, n-1):
            a[i][j] = temp3[i]

    return a


def core(x, xoffset, dt, params):
    yobs = f(propagate(x, dt, params))
    ypred = f(propagate(x+xoffset, dt, params))

    xi = yobs - ypred

    hello = np.zeros(20, 1)

    for i in range(0, 10):  # stopping criteria, should be changed to be based on deltaX
        hello[i] = la.norm(xi)

        b = -__derivative()
        c = np.matmul(b.transpose(), b)
        d = -np.matmul(b.transpose(), xi)

        deltax = np.matmul(la.inv(c), d)

        xnew = x + deltax

        ypred = f(propagate(xnew, dt, params))
        x = xnew
        xi = yobs - ypred

    return xnew
