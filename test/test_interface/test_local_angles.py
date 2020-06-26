import numpy as np
import math
from src.interface.local_angles import *
import pytest
import astropy.units as u
from astropy.time import Time
from src.enums import Frames
from src.dto import PropParams
import mockito
from mockito import patch, when
from src.interface import local_angles as la
from test import xcompare

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


@pytest.mark.parametrize("lla, expected", [([90 * u.deg, 0 * u.deg, 0 * u.km],
                                            np.array([np.arccos(3/norm)*rtd, 90 - np.arctan2(-1, 2) * rtd])),
                                           ([0 * u.deg, 0 * u.deg, 0 * u.km],
                                            np.array([np.arccos(1/norm)*rtd, 90 - np.arctan2(3, 2) * rtd])),
                                           ([0 * u.deg, 90 * u.deg, 0 * u.km],
                                            np.array([np.arccos(2/norm)*rtd, 90 - np.arctan2(3, -1) * rtd])),
                                           ([-90 * u.deg, 0 * u.deg, 0 * u.km],
                                            np.array([np.arccos(-3/norm)*rtd, 90 - np.arctan2(1, 2) * rtd]))])
def test_local_angles(lla, expected):
    rr = np.array([1, 2, 3])
    actual = local_angles(rr, lla)
    assert np.allclose(expected, actual)


def test_get_local_angles_for_state_prop():
    x = np.array([1, 2, 3, 4, 5, 6])
    obs_pos_lla = [1, 2, 3]
    obs_frame = Frames.LLA
    epoch_i = Time(2454283.0, format="jd", scale="tdb")
    epoch_f = epoch_i + 1 * u.day
    n = 0
    params = PropParams(epoch_i)

    mocked_angles1 = np.array([1, 2])
    mocked_angles2 = np.array([2, 3])
    mocked_state1 = np.array([7, 8, 9, 10, 11, 12])
    mocked_state2 = np.array([13, 14, 15, 16, 17, 18])
    expected = [[1, 2, epoch_i], [2, 3, epoch_f]]

    obs_pos_ecef = np.array([4, 5, 6])

    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(la).lla_to_ecef(obs_pos_lla).thenReturn(obs_pos_ecef)
        when(la).ecef_to_eci(obs_pos_ecef, epoch_i).thenReturn(np.zeros(3))
        when(la).ecef_to_eci(obs_pos_ecef, epoch_f).thenReturn(np.zeros(3))
        when(la).state_propagate(x, epoch_i, params).thenReturn(mocked_state1)
        when(la).state_propagate(x, epoch_f, params).thenReturn(mocked_state2)
        when(la).local_angles(mocked_state1[0:3], obs_pos_lla).thenReturn(mocked_angles1)
        when(la).local_angles(mocked_state2[0:3], obs_pos_lla).thenReturn(mocked_angles2)
        actual = get_local_angles_for_state_propagation(x, params, epoch_f, n, obs_pos_lla, obs_frame)
    assert actual == expected
