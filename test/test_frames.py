import astropy.units as u
from src.util import *
import numpy as np
from src.frames import lla_to_ecef, ecef_to_lla, ecef_to_eci, eci_to_ecef
import pytest


@pytest.mark.parametrize("input, expected", [([90 * u.deg, 0 * u.deg, 0 * u.km], np.array([0, 0, 6356.75231])),
                                             ([0 * u.deg, 0 * u.deg, 0 * u.km], np.array([6378.137, 0, 0]))])
def test_lla_to_ecef(input, expected):
    actual = lla_to_ecef(input)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("input, expected", [(np.array([0, 0, 6356.75231]), [90 * u.deg, 0 * u.deg, -4.2451791e-6 * u.km]),
                                             (np.array([6378.137, 0, 0]), [0 * u.deg, 0 * u.deg, 0 * u.km])])
def test_ecef_to_lla(input, expected):
    actual = ecef_to_lla(input)
    for i in range(3):
        assert expected[i].unit == actual[i].unit
        thing = np.array(expected[i].value)
        thong = np.array(actual[i].value)
        assert np.allclose(thing, thong)


@pytest.mark.parametrize("input", [(np.array([0, 0, 6356.75231])),
                                    (np.array([6378.137, 0, 0]))])
def test_eci_to_ecef_and_back(input):
    epoch = Time("2018-08-17 12:05:50", scale="tdb")
    middle = eci_to_ecef(input, epoch)
    actual = ecef_to_eci(middle, epoch)
    assert np.linalg.norm(actual - input) < 1e-7
#   To supplement the there and back. Ground tracks will be observed in a verification file.
