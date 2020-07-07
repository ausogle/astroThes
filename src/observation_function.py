import numpy as np
import numpy.linalg as la
from src.dto import Observation
from src.enums import Frames


def y(x: np.ndarray, observation: Observation):
    """
    This function serves as a prediction function. It is used to describe the right ascension and declination of an
    observed satellite from an observer. Math is done in ECI Frame
    :param x: State Vector of satellite in ECI frame.
    :param observation: Parameters relevant to observation. Includes epoch, frame, and location of observation/observer.
    """
    r_obj = x[0:3]
    assert observation.frame == Frames.ECI
    rr = r_obj - observation.position
    alpha, dec = get_ra_and_dec(rr)
    return np.array([alpha, dec])


def get_ra_and_dec(rr: np.ndarray) -> np.ndarray:
    """
    Prediction function. Determines observational angles (WHICH) of the sateliite from observer.
    :param rr: Position of the spacecraft relative to observer
    """
    alpha = np.arctan2(rr[1], rr[0]) * 180 / np.pi
    dec = 90 - np.arccos(rr[2]/la.norm(rr)) * 180 / np.pi
    return np.array([alpha, dec])
