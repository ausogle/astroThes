from src.interface.tle_dto import TLE
import numpy as np
from src.dto import PropParams
from astropy.time import Time


def test_tle_to_state():
    tle_string = """
ISS (ZARYA)
1 25544U 98067A   20171.48973192 -.00009734  00000-0 -16612-3 0  9998
2 25544  51.6439 337.0700 0002381  67.4024  37.3218 15.49440074232376
"""
    tle = TLE.from_lines(tle_string)
    sat = tle.to_orbit()
    x_expected = np.concatenate([sat.r.value, sat.v.value])

    epoch_yr = 2020
    epoch_day = 171.48973192
    epoch_expected = Time(epoch_yr + epoch_day / 365.25, format="decimalyear", scale="utc")

    x_actual, epoch_out = tle.to_state()
    assert np.array_equal(x_expected, x_actual)
    assert epoch_out == epoch_expected


# def test_tle_to_string_loop():
#     tle_in = """
# ISS (ZARYA)
# 1 25544U 98067A   20171.48973192 -.00009734  00000-0 -16612-3 0  9998
# 2 25544  51.6439 337.0700 0002381  67.4024  37.3218 15.49440074232376
# """
#     tle = TLE.from_lines(tle_in)
#     tle_out = tle.to_string()
#     assert tle_in == tle_out
