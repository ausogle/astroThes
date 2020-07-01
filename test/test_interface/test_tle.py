from tletools import TLE
import numpy as np
from src.interface.tle import tle_to_state, state_to_tle
from src.dto import PropParams


def test_tle_to_state():
    tle_string = """
ISS (ZARYA)
1 25544U 98067A   20171.48973192 -.00009734  00000-0 -16612-3 0  9998
2 25544  51.6439 337.0700 0002381  67.4024  37.3218 15.49440074232376
"""
    tle_lines = tle_string.strip().splitlines()
    tle = TLE.from_lines(*tle_lines)
    sat = tle.to_orbit()
    x_expected = np.concatenate([sat.r.value, sat.v.value])
    epoch_expected = sat.epoch
    x_actual, prop_params = tle_to_state(tle_string)
    assert np.array_equal(x_expected, x_actual)
    assert prop_params.epoch == epoch_expected


def test_state_to_tle():
    tle_in = """
ISS (ZARYA)
1 25544U 98067A   20171.48973192 -.00009734  00000-0 -16612-3 0  9998
2 25544  51.6439 337.0700 0002381  67.4024  37.3218 15.49440074232376
"""
    tle_lines = tle_in.strip().splitlines()
    tle = TLE.from_lines(*tle_lines)
    sat = tle.to_orbit()
    x = np.concatenate([sat.r.value, sat.v.value])
    params = PropParams(sat.epoch)

    tle_expected = """ISS (ZARYA)
1 25544U 98067A   20170.14036771 -.00009734  00000-0 -16612-3 0  9993
2 25544  51.6439 337.0699 0002381  67.4024  37.3385 15.49440074232378"""

    tle_actual = state_to_tle(tle_in, x, params)
    assert tle_actual == tle_expected
