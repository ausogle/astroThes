import numpy as np
from src import observation_function
from src.observation_function import *
from src.enums import Frames
from src.dto import ObsParams
import mockito
import pytest
import mockito
from mockito import patch, when
from test import xcompare


@pytest.mark.parametrize("rr, expected", [(np.array([100, 100, 0]), np.array([45, 0])),
                                          (np.array([100, 0, 0]), np.array([0, 0])),
                                          (np.array([0, 100, 0]), np.array([90, 0])),
                                          (np.array([0, 0, -100]), np.array([0, -90]))])
def test_get_ra_and_dec(rr, expected):
    actual = get_ra_and_dec(rr)
    assert np.array_equal(actual, expected)


def test_f_with_eci_frame():
    position = np.array([100, 100, 100])
    obs_params = ObsParams(position, Frames.ECI, None)
    x = np.array([100, 200, 100])
    expected = np.array([90, 0])
    actual = y(x, obs_params)
    assert np.array_equal(expected, actual)


def test_y_with_ecef_frame():
    position = np.array([0, 0, 0])
    obs_params = ObsParams(position, Frames.ECEF, None)
    x = np.array([100, 50, 50, 0, 0, 0])
    r_obj = x[0:3]
    expected = (66, 66)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(observation_function).ecef_to_eci(position, None).thenReturn(r_obj)
        when(observation_function).get_ra_and_dec(np.zeros(3)).thenReturn(expected)
        actual = y(x, obs_params)
    assert np.array_equal(actual, expected)
