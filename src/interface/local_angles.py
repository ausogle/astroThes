import numpy as np
from typing import List


def local_angles(rr: np.ndarray, lla: List):
    """
    Gives local azimuth and zenith angles for a satellite with respect to an observer's position on the Earth. Theta is
    the zenith angel, measured from directly above. The azimuthal angle is phi, measured from local North, where
    positive indicates Eastward or clockwise.

    :param rr: observational difference vector in ECEF frame. Units [km]
    :param lla: List of [lat, long, altitude]. Units [deg, deg, km]
    """
    rot_mat = rotation_matrix(lla[0].value, lla[1].value)
    local_sky = rot_mat.T @ rr

    elevation = 90 - np.arccos(local_sky[2] / np.linalg.norm(local_sky)) * 180 / np.pi #Changed for elevation not declination. Not included in comments above.
    azimuth = 90 - np.arctan2(-local_sky[0], local_sky[1]) * 180 / np.pi

    return azimuth, elevation


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
