import numpy as np
from numpy import linalg as la
from Ffun import f
from propagator import propagate


def __derivative():
    return np.zeros((6, 2))


def core(x, xoffset, dt, params):
    yobs = f(propagate(x, dt, params))
    ypred = f(propagate(x+xoffset, dt, params))

    xi = yobs - ypred

    hello = np.zeros(20, 1)

    for i in range(0, 10):  # stopping criteria
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
