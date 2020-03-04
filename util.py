import numpy as np
from numpy import linalg as la
import math


def rv_to_oe(rr, vv):
    a = __get_a(rr, vv)
    e = __get_e(rr, vv)
    i = __get_i(rr, vv)
    raan = __get_raan(rr, vv)
    omega = __get_omega(rr, vv)
    theta = __get_theta(rr, vv)
    return np.array([a, e, i, raan, omega, theta])


def __get_a(rr, vv):
    mu = 39860.4418
    r = la.norm(rr)
    v = la.norm(vv)

    eps = v * v / 2 - mu / r
    a = -mu / (2 * eps)
    return a


def __get_e(rr, vv):
    mu = 39860.4418
    r = la.norm(rr)

    hh = np.cross(rr, vv)

    ee = np.cross(vv / mu, hh) - rr / r
    e = la.norm(ee)
    return e


def __get_i(rr, vv):
    hh = np.cross(rr, vv)
    h = la.norm(hh)

    i = math.acos(np.dot(hh, np.array([0, 0, 1]))/h)
    return i


def __get_raan(rr, vv):
    hh = np.cross(rr, vv)
    nn = np.cross(np.array([0, 0, 1]), hh)
    n = la.norm(nn)

    quantity = np.dot(nn, np.array([0, 1, 0])) / n
    if n == 0:
        quantity = 1

    if np.dot(nn, np.array([0, 1, 0])) >= 0:
        raan = math.acos(quantity)
    else:
        raan = 2 * math.pi - math.acos(quantity)
    return raan


def __get_omega(rr, vv):
    mu = 39860.4418
    r = la.norm(rr)
    hh = np.cross(rr, vv)
    h = la.norm(hh)
    ee = np.cross(vv / mu, hh) - rr / r
    e = la.norm(ee)
    nn = np.cross(np.array([0, 0, 1]), hh)
    n = la.norm(nn)

    quantity = np.dot(nn, ee) / (n * e)
    if n == 0:
        quantity = -1
    if np.dot(ee, np.array([0, 0, 1])) >= 0:
        omega = math.acos(quantity)
    else:
        omega = 2 * math.pi - math.acos(quantity)
    return omega


def __get_theta(rr, vv):
    mu = 39860.4418
    r = la.norm(rr)
    hh = np.cross(rr, vv)
    ee = np.cross(vv / mu, hh) - rr / r
    e = la.norm(ee)

    if np.dot(rr, vv) >= 0:
        theta = math.acos(np.dot(ee, rr)/(e*r))
    else:
        theta = 2*math.pi-math.acos(np.dot(ee, rr)/(e*r))
    return theta
