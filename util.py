import numpy as np
from numpy import linalg as la
import math


class ObsParams:
    def __init__(self, obs_loc, obs_frame, epoch_f):
        self.obs_loc = obs_loc
        self.obs_frame = obs_frame
        self.epoch_f = epoch_f


class PropParams:
    def __init__(self, dt, epoch_f):
        self.dt = dt
        self.epoch = epoch_f
        self.perturbations = {}

    def add_perturbation(self, name, perturbation):
        self.perturbations[name] = perturbation


class J2:
    def __init__ (self, J2, R):
        self.J2 = J2
        self.R = R


class J3:
    def __init__ (self, J3, R):
        self.J3 = J3
        self.R = R


class Drag:
    def __init__(self, R, C_D, A, m, H0, rho0):
        self.R = R
        self.C_D = C_D
        self.A = A
        self.m = m
        self.H0 = H0
        self.rho0 = rho0


class ThirdBody:
    def __init__(self, k_third, third_body):
        self.k_third = k_third
        self.third_body = third_body


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
