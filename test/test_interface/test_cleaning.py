from src.dto import ObsParams
from src.enums import Frames
import mockito
from test.test_core import xcompare
from mockito import patch, when
import astropy.units as u
from src.interface.cleaning import convert_obs_params_from_lla_to_ecef, verify_units
from src.interface import cleaning
import numpy as np
import math


def test_convert_obs_params_from_lla_to_ecef():
    input_pos = [0 * u.deg, 0 * u.deg, 0 * u.km]
    output_pos = np.array([0, 0, 0])
    input = ObsParams(input_pos, Frames.LLA, None)
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(cleaning).lla_to_ecef(input_pos).thenReturn(np.array(output_pos))
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
    assert expected_outcome.frame == actual_outcome.frame
    for i in range(3):
        assert expected_outcome.position[i] == actual_outcome.position[i]


def test_verify_units_lla():
    position = [math.pi*2 * u.rad, math.pi*2 * u.rad, 1000 * u.m]
    obs_params_in = ObsParams(position, Frames.LLA, None)
    expected_position = [360 * u.deg, 360 * u.deg, 1 * u.km]
    expected_outcome = ObsParams(expected_position, Frames.LLA, None)
    actual_outcome = verify_units(obs_params_in)
    assert expected_outcome.frame == actual_outcome.frame
    for i in range(3):
        assert expected_outcome.position[i] == actual_outcome.position[i]
