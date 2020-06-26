import numpy as np
from src.state_propagator import state_propagate
from src.dto import PropParams
from src.enums import Frames
from src.frames import lla_to_ecef, ecef_to_lla, ecef_to_eci
from typing import List
from astropy.time import Time
from astropy.coordinates import Angle


def local_angles(rr: np.ndarray, lla: List) -> np.ndarray:
    """
    Gives local azimuth and zenith angles for a satellite with respect to an observer's position on the Earth. Theta is
    the zenith angel, measured from directly above. The azimuthal angle is phi, measured from local North, where
    positive indicates Eastward or clockwise.

    :param rr: observational difference vector in ECEF frame. Units [km]
    :param lla: List of [lat, long, altitude]. Units [deg, deg, km]
    """
    rot_mat = rotation_matrix(lla[0].value, lla[1].value)
    local_sky = rot_mat.T @ rr

    theta = np.arccos(local_sky[2] / np.linalg.norm(local_sky)) * 180 / np.pi
    phi = 90 - np.arctan2(-local_sky[0], local_sky[1]) * 180 / np.pi

    return np.array([theta, phi])


def ry(lat: float) -> np.ndarray:
    """
    Gives rotation matrix to rotate around the y-axis in ECEF frame to align the z-axis with the observers latitude

    :param lat: Latitude of the observer. Units [deg]
    """
    angle = -lat * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    a = np.array([[-s, 0, c], [0, 1, 0], [-c, 0, -s]])
    return a


def rz(lon: float) -> np.ndarray:
    """
    Gives rotation matrix to rotate around the z-axis in ECEF from to align x-axis with observers longitude

    :param lon: Longitude of the observer. Units [deg]
    """
    angle = lon * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    a = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return a


def rotation_matrix(lat, lon):
    """
    Build rotation matrix from ECEF z-axis along direction of observer. This transpose of this matrix converts an
    observational difference vector into an x, y, z coordinate system x and y are perpendicular components of the
    observation vector and z is the parallel component. Z gives zenith angle while x,y give azimuth angle. Notably,
    in this frame x points south and y points east.

    :param lat: Latitude of the observer
    :param lon: Longitude of the observer
    """
    return rz(lon) @ ry(lat)


def get_local_angles_for_state_propagation(x: np.ndarray, prop_params: PropParams, epoch_f: Time, n: int,
                                           obs_location, obs_frame: Frames):
    """
    This function returns a list of [theta, phi, Time] from initial epoch in prop_params to final epoch, with an
    n points between those two epochs.

    :param x: State vector at initial epoch
    :param prop_params: Parameters relevant to propagation. Includes initial epoch
    :param epoch_f: Epoch of final desired time
    :param n: Number of desired points between initial and final epoch (Does not include those two)
    :param obs_location: Observer location. Accepts list for LLA or np.array for ECEF
    :param obs_frame: Frame observer location is in. Accepts Frames.LLA or Frames.ECEF
    """
    if obs_frame == Frames.LLA:
        obs_pos_lla = obs_location
        obs_pos_ecef = lla_to_ecef(obs_location)
    else:
        assert obs_frame == Frames.ECEF
        obs_pos_lla = ecef_to_lla(obs_location)
        obs_pos_ecef = obs_location

    epoch_i = prop_params.epoch
    dt = (epoch_f - epoch_i)/ (n+1)
    output = []
    for i in range(0, n + 2):
        desired_epoch = epoch_i + dt * i
        obs_pos_eci = ecef_to_eci(obs_pos_ecef, desired_epoch)
        obj_pos_eci = state_propagate(x, desired_epoch, prop_params)
        rr = obj_pos_eci[0:3] - obs_pos_eci
        angles = local_angles(rr, obs_pos_lla)
        output.append([angles[0], angles[1], desired_epoch])
    return output
