import math
import numpy as np
from numpy import linalg as la
from typing import List
from util import ObsParams


def f(rr: np.ndarray, obs_params: ObsParams) -> np.ndarray:
    """
    Prediction function. Should simulate the angles a telescope would measure if located at the center of the Earth.
    Will be expanded.
    :param rr: Position of the spacecraft, in ECI frame
    :param obs_params: Object that contains all supplemental information such as the observer's location, the frame
    that was measured in, as well as the time of measurement in case a transformation from ECEF to ECI is required.
    :return: Returns azimuthal and horizontal angles a user would observe if the satellite was at location rr
    """
    alpha = math.atan2(rr[1], rr[0]) * 180/math.pi
    dec = 90 - math.acos(rr[2]/la.norm(rr))*180/math.pi
    return np.array([alpha, dec])

