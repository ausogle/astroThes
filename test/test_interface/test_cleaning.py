from src.dto import Observation
from src.enums import Frames
import mockito
from test.test_core import xcompare
from mockito import patch, when
import astropy.units as u
from src.interface.cleaning import convert_obs_from_lla_to_ecef, convert_obs_from_lla_to_eci, \
    convert_obs_from_ecef_to_eci, verify_locational_units
from src.interface import cleaning
import numpy as np
import math


def test_convert_obs_params_from_lla_to_ecef():
    input_pos = [0 * u.deg, 0 * u.deg, 0 * u.km]
    output_pos = np.array([0, 0, 0])
    epoch = None
    obs_values = None
    obs_type = None

    input = Observation(input_pos, Frames.LLA, epoch, obs_values, obs_type)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(cleaning).lla_to_ecef(input_pos).thenReturn(np.array(output_pos))
    expected = Observation(output_pos, Frames.ECEF, epoch, obs_values, obs_type)
    actual = convert_obs_from_lla_to_ecef(input)
    assert np.array_equal(expected.position, actual.position)
    assert expected.frame == actual.frame


def test_convert_obs_params_from_lla_to_eci():
    input_pos = [0 * u.deg, 0 * u.deg, 0 * u.km]
    output_pos = np.array([0, 0, 0])
    epoch = None
    obs_values = None
    obs_type = None

    input = Observation(input_pos, Frames.LLA, epoch, obs_values, obs_type)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(cleaning).lla_to_ecef(input_pos).thenReturn("Nothing")
        when(cleaning).ecef_to_eci("Nothing", epoch).thenReturn(np.array(output_pos))
    expected = Observation(output_pos, Frames.ECI, epoch, obs_values, obs_type)
    actual = convert_obs_from_lla_to_eci(input)
    assert np.array_equal(expected.position, actual.position)
    assert expected.frame == actual.frame


def test_convert_obs_from_ecef_to_eci():
    input_pos = [1 * u.km, 2 * u.km, 3 * u.km]
    output_pos = np.array([0, 0, 0])
    epoch = None
    obs_values = None
    obs_type = None

    input = Observation(input_pos, Frames.ECEF, epoch, obs_values, obs_type)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(cleaning).ecef_to_eci(input_pos, epoch).thenReturn(np.array(output_pos))
    expected = Observation(output_pos, Frames.ECI, epoch, obs_values, obs_type)
    actual = convert_obs_from_ecef_to_eci(input)
    assert np.array_equal(expected.position, actual.position)
    assert expected.frame == actual.frame


def test_verify_units_spacial():
    position = [1000 * u.m, 1000 * u.m, 1000 * u.m]
    epoch = None
    obs_values = None
    obs_type = None

    obs_params_in = Observation(position, Frames.ECI, epoch, obs_values, obs_type)
    expected_position = [1 * u.km, 1 * u.km, 1 * u.km]
    expected_outcome = Observation(expected_position, Frames.ECI, epoch, obs_values, obs_type)
    actual_outcome = verify_locational_units(obs_params_in)
    assert expected_outcome.frame == actual_outcome.frame
    for i in range(3):
        assert expected_outcome.position[i] == actual_outcome.position[i]


def test_verify_units_lla():
    position = [math.pi*2 * u.rad, math.pi*2 * u.rad, 1000 * u.m]
    epoch = None
    obs_values = None
    obs_type = None

    obs_params_in = Observation(position, Frames.LLA, epoch, obs_values, obs_type)
    expected_position = [360 * u.deg, 360 * u.deg, 1 * u.km]
    expected_outcome = Observation(expected_position, Frames.LLA, epoch, obs_values, obs_type)
    actual_outcome = verify_locational_units(obs_params_in)
    assert expected_outcome.frame == actual_outcome.frame
    for i in range(3):
        assert expected_outcome.position[i] == actual_outcome.position[i]
