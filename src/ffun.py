import math
import numpy as np
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
from astropy.time import Time
import numpy.linalg as la
from src.dto import ObsParams
from src.enums import Frames


def f(x: np.ndarray, obs_params: ObsParams):
    """
    This function serves as a prediction function. It is used to describe the right ascension and declination of an
    observed satellite from an observer. Math is done in ECI/ECEF Frame
    :param x: State Vector of satellite in ECI frame.
    :param obs_params: Parameters relevant to observation. Includes epoch, frame, and location of observation/observer.
    """
    r_obj = x[0:3]
    if obs_params.frame == Frames.ECI:
        rr = r_obj - obs_params.position
    else:
        r_obs = obs_params.position
        r_obj_ecef = eci_to_ecef(r_obj, obs_params.epoch)
        rr = r_obj_ecef - r_obs
    alpha, dec = get_ra_and_dec(rr)
    return np.array([alpha, dec])


def get_ra_and_dec(rr: np.ndarray) -> np.ndarray:
    """
    Prediction function. Determines observational angles of the sateliite from observer.
    :param rr: Position of the spacecraft relative to observer
    """
    alpha = math.atan2(rr[1], rr[0]) * 180/math.pi
    dec = 90 - math.acos(rr[2]/la.norm(rr))*180/math.pi
    return np.array([alpha, dec])


def eci_to_ecef(r: np.ndarray, time: Time):
    """
    Converts coordinates in Earth Centered Inertial frame to Earth Centered Earth Fixed.
    :param r: position of satellite in ECI frame. Units [km]
    :param time: Time of observation
    """
    gcrs = GCRS(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km), obstime=time)
    itrs = gcrs.transform_to(ITRS(obstime=time))
    x_ecef = itrs.x.value
    y_ecef = itrs.y.value
    z_ecef = itrs.z.value
    r = np.array([x_ecef, y_ecef, z_ecef])
    return r



