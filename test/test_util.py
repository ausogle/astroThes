from src.dto import ObsParams
from src.enums import Frames
import astropy.units as u
from src.util import *
import math
import numpy as np
from test.test_core import xcompare
import mockito
from mockito import when, patch
from src import util


def test_lla_to_ecef_north_pole():
    input = [90 * u.deg, 0 * u.deg, 0 * u.km]
    expected = np.array([0, 0, 6356.75231])
    actual = lla_to_ecef(input)
    assert np.allclose(actual, expected)


def test_lla_to_ecef_greenwich():
    input = [0 * u.deg, 0 * u.deg, 0 * u.km]
    expected = np.array([6378.137, 0, 0])
    actual = lla_to_ecef(input)
    assert np.array_equal(actual, expected)


def test_convert_obs_params_from_lla_to_ecef():
    input_pos = [0 * u.deg, 0 * u.deg, 0 * u.km]
    output_pos = np.array([0, 0, 0])
    input = ObsParams(input_pos, Frames.LLA, None)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(util).lla_to_ecef(input_pos).thenReturn(np.array(output_pos))
    expected = ObsParams(output_pos, Frames.ECEF, None)
    actual = convert_obs_params_from_lla_to_ecef(input)
    assert np.array_equal(expected.position, actual.position)
    assert expected.frame == actual.frame


def test_verify_units_spacial():
    position = [1000 * u.m, 1000 * u.m, 1000 * u.m]
    obs_params_in = ObsParams(position, Frames.ECI, None)
    expected_position = [1 * u.km, 1 * u.km, 1 * u.km]
    expected_outcome = ObsParams(expected_position, Frames.ECI, None)
    actual_outcome = verify_units(obs_params_in)
    assert(expected_outcome, actual_outcome)


def test_verify_units_lla():
    position = [math.pi*2 * u.rad, math.pi*2 * u.rad, 1000 * u.m]
    obs_params_in = ObsParams(position, Frames.LLA, None)
    expected_position = [0 * u.rad, 0 * u.rad, 1 * u.km]
    expected_outcome = ObsParams(expected_position, Frames.ECI, None)
    actual_outcome = verify_units(obs_params_in)
    assert(expected_outcome, actual_outcome)
