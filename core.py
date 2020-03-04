import numpy as np
from numpy import linalg as la
import math


def __derivative():
    return np.zeros((6, 2))


def core(X, Xoffset, dt, delta, params):
    # yobs = F(propagate(X,m,dt,dt/1000));
    # ypred = F(propagate(X+Xoffset,m,dt,dt/1000));

    yobs = np.zeros((2, 1))
    ypred = np.zeros((2, 1))

    Xi = yobs - ypred

    hello = np.zeros(20, 1)

    for i in range(0, 10):  # stopping criteria
        hello[i] = la.norm(Xi)

        B = -__derivative()
        C = np.matmul(B.transpose(), B)
        D = -np.matmul(B.transpose(), Xi)

        deltaX = np.matmul(la.inv(C), D)

        Xnew = X + deltaX

        # ypred = F(propagate(Xnew,m,dt,dt/1000))
        X = Xnew
        Xi = yobs - ypred
