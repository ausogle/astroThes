import numpy as np
import math
mu = np.float64(398600.4418)    # gravitational constant of the Earth km^3/s^2


def generate_earth_surface():
    r = 6378
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def get_a(x):
    rr = x[0:3]
    vv = x[3:6]
    v = np.linalg.norm(vv)
    r = np.linalg.norm(rr)
    eps = v*v/2 - (mu/r)
    a = -mu/(2*eps)
    return a


def get_period(x):
    a = get_a(x)
    t = 2*np.pi*math.sqrt(a*a*a/mu)
    return t


def get_e(x):
    rr = x[0:3]
    vv = x[3:6]
    r = np.linalg.norm(rr)
    hh = np.cross(rr, vv)
    ee = np.cross(vv/mu, hh) - (rr/r)
    e = np.linalg.norm(ee)
    return e
