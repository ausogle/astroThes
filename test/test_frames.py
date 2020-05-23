import astropy.units as u
from src.util import *
import numpy as np
from src.frames import lla_to_ecef


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



