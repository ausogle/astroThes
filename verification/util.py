import numpy as np
import math
from src.dto import PropParams, Observation
from src.enums import Angles, Frames
from src.state_propagator import state_propagate
from src.frames import lla_to_ecef, ecef_to_eci
from src.observation_function import y
from src.constants import mu
import astropy.units as u


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


def get_satellite_position_over_time(x, init_epoch, epochs):
    r = np.zeros((len(epochs), 3))
    r[0] = x[0:3]
    params = PropParams(init_epoch)
    for i in range(1, len(epochs)):
        x_temp = state_propagate(x, epochs[i], params)
        r[i] = x_temp[0:3]
    return r, epochs


sigma_theta = .003


def build_observations(x, prop_params, obs_pos, frame, epochs, sigmas=np.array([sigma_theta, sigma_theta])):
    output = []
    temp = []
    assert frame == Frames.LLA
    for k in range(len(epochs)):
        epoch = epochs[k]
        x_k = state_propagate(x, epoch, prop_params)
        pos = ecef_to_eci(lla_to_ecef(obs_pos), epoch)
        temp.append(Observation(pos, Frames.ECI, epoch, None, Angles.Local, sigmas))
        obs_values = y(x_k, temp[k])
        output.append(Observation(pos, Frames.ECI, epoch, obs_values, Angles.Local, sigmas))
    return output


def build_epochs(epoch, stepsize, steps):
    epochs = []
    for i in range(steps):
        epochs.append(epoch + i * stepsize)
    return epochs


def build_noisy_observations(x, prop_params, obs_pos, frame, epochs, noise=1/60):
    observations = build_observations(x, prop_params, obs_pos, frame, epochs, sigmas=np.array([noise, noise]))
    for obs in observations:
        obs.obs_values = obs.obs_values + np.random.rand(2) * noise
    return observations
