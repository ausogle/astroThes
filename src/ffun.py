import math
import numpy as np
import numpy.linalg as la
from src.dto import ObsParams
from src.enums import Frames
from src.frames import eci_to_ecef


# def f(x, obs_params):
#     rr = x[0:3]
#     alpha = math.atan2(rr[1], rr[0]) * 180 / math.pi
#     dec = 90 - math.acos(rr[2] / la.norm(rr)) * 180 / math.pi
#     return np.array([alpha, dec])


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
    Prediction function. Determines observational angles (WHICH) of the sateliite from observer.
    :param rr: Position of the spacecraft relative to observer
    """
    alpha = math.atan2(rr[1], rr[0]) * 180/math.pi
    dec = 90 - math.acos(rr[2]/la.norm(rr))*180/math.pi
    return np.array([alpha, dec])






