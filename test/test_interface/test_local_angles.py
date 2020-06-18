import numpy as np
import math
from src.interface.local_angles import rotation_matrix, local_angles
import pytest
import astropy.units as u

rt = math.sqrt(2)


@pytest.mark.parametrize("lla, expected", [([0 * u.deg, 0 * u.deg, 800 * u.km], np.array([-3, 2, 1])),
                                           ([0 * u.deg, 90 * u.deg, 800 * u.km], np.array([-3, -1, 2])),
                                           ([90 * u.deg, 0 * u.deg, 800 * u.km], np.array([1, 2, 3])),
                                           ([45 * u.deg, 45 * u.deg, 800 * u.km], np.array([(3-3*rt)/2, 1/rt, (3+3*rt)/2]))])
def test_r(lla, expected):
    rr = np.array([1, 2, 3])
    rot_mat = rotation_matrix(lla[0].value, lla[1].value)
    actual = rot_mat.T @ rr
    assert np.allclose(expected, actual)


norm = np.linalg.norm(np.array([1, 2, 3]))
rtd = 180 / np.pi


@pytest.mark.parametrize("lla, expected", [([90 * u.deg, 0 * u.deg, 10 * u.km],
                                            np.array([np.arccos(3/norm)*rtd, 90 - np.arctan2(-1, 2) * rtd])),
                                           ([0 * u.deg, 0 * u.deg, 10 * u.km],
                                            np.array([np.arccos(1/norm)*rtd, 90 - np.arctan2(3, 2) * rtd])),
                                           ([0 * u.deg, 90 * u.deg, 10 * u.km],
                                            np.array([np.arccos(2/norm)*rtd, 90 - np.arctan2(3, -1) * rtd])),
                                           ([-90 * u.deg, 0 * u.deg, 10 * u.km],
                                            np.array([np.arccos(-3/norm)*rtd, 90 - np.arctan2(1, 2) * rtd]))])
def test_local_angles(lla, expected):
    rr = np.array([1, 2, 3])
    actual = local_angles(rr, lla)
    assert np.allclose(expected, actual)

