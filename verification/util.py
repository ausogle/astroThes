import numpy as np
import math
from src.dto import PropParams, Observation
from src.enums import Angles, Frames
from src.state_propagator import state_propagate
from src.frames import lla_to_ecef, ecef_to_eci, eci_to_eci_angles, eci_to_icrs_angles
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


def get_satellite_position_over_time(x, epoch, tf, dt) -> np.matrix:
    t = np.arange(0, tf, dt)            #Units of s. No astropy unit attached, just a scalar
    r = np.zeros((len(t), 3))
    prop_params = PropParams(epoch)
    r[0] = x[0:3]
    for i in range(1, len(t)):
        desired_epoch = epoch + t[i] * u.s
        x = state_propagate(x, desired_epoch, prop_params)
        r[i] = x[0:3]
        prop_params.epoch = desired_epoch
    return r


sigma_theta = .003


def build_y_observations(x, prop_params, obs_pos, frame, epochs, sigmas=np.array([sigma_theta, sigma_theta])):
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


def build_eci_observations(x, prop_params, obs_pos, frame, epochs):
    output = []
    assert frame == Frames.LLA
    for k in range(len(epochs)):
        epoch = epochs[k]
        x_k = state_propagate(x, epoch, prop_params)
        pos = ecef_to_eci(lla_to_ecef(obs_pos), epoch)
        output.append(eci_to_eci_angles(x_k[0:3] - pos, epoch))
    return output


def build_icrs_observations(x, prop_params, obs_pos, frame, epochs):
    output = []
    assert frame == Frames.LLA
    for k in range(len(epochs)):
        epoch = epochs[k]
        x_k = state_propagate(x, epoch, prop_params)
        pos = ecef_to_eci(lla_to_ecef(obs_pos), epoch)
        output.append(eci_to_icrs_angles(x_k[0:3] - pos, epoch))
    return output


def build_epochs(epoch, stepsize, steps):
    epochs = []
    for i in range(steps):
        epochs.append(epoch + i * stepsize)
    return epochs


def build_noisy_observations(x, prop_params, obs_pos, frame, epochs, noise=1/60):
    observations = build_y_observations(x, prop_params, obs_pos, frame, epochs, sigmas=np.array([noise, noise]))
    for obs in observations:
        obs.obs_values = obs.obs_values + np.random.rand(2) * noise
    return observations
