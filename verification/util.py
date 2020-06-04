import numpy as np
import math
from src.dto import PropParams
from src.propagator import propagate
from src.constants import mu


def generate_earth_surface():
    """
    Generates x,y,z coordinates for a perfect sphere representing the Earth. To be used in plot_surface()
    """
    r = 6378
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def get_a(x):
    """
    Returns semi-major axis of an orbit given the state x = [r v]. Unit: [km]
    """
    rr = x[0:3]
    vv = x[3:6]
    v = np.linalg.norm(vv)
    r = np.linalg.norm(rr)
    eps = v*v/2 - (mu.value/r)
    a = -mu.value/(2*eps)
    return a


def get_period(x):
    """
    Returns the period of an orbit given the state x = [r v]. Unit: [s]
    """
    a = get_a(x)
    t = 2*np.pi*math.sqrt(a*a*a/mu.value)
    return t


def get_e(x):
    """
    Returns the eccentricity of an orbit given the state x = [r v]
    """
    rr = x[0:3]
    vv = x[3:6]
    r = np.linalg.norm(rr)
    hh = np.cross(rr, vv)
    ee = np.cross(vv/mu, hh) - (rr/r)
    e = np.linalg.norm(ee)
    return e


def get_satellite_position_over_time(x, epoch, tf, dt) -> np.matrix:
    t = np.arange(0, tf, dt)
    r = np.zeros((len(t), 3))
    prop_params = PropParams(dt, epoch)
    for i in range(0, len(t)):
        r[i] = x[0:3]
        x = propagate(x, prop_params)
        prop_params.epoch = prop_params.epoch + prop_params.dt
    return r
