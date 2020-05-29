from src import core
from src.core import *
from src.interface.cleaning import convert_obs_params_from_lla_to_ecef
import mockito
from mockito import when, patch
import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from src.enums import Frames
from verification.util import get_period
from test import xcompare


def test_convergence():
    r = [66666, 0, 0]
    v = [0, -2.644, 1]
    x = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    period = get_period(x)
    tf = period / 2
    epoch = Time(2454283.0, format="jd", scale="tdb")
    x_offset = np.array([100, 50, 10, .01, .01, .03])
    obs_params = ObsParams([0 * u.deg, 0 * u.deg, 800 * u.km], Frames.LLA, epoch)
    obs_params = convert_obs_params_from_lla_to_ecef(obs_params)
    prop_params = PropParams(tf, epoch)
    yobs = y(propagate(x + x_offset, prop_params), obs_params)
    x_alg = milani(x, yobs, obs_params, prop_params)
    yalg = y(propagate(x_alg, prop_params), obs_params)
    assert np.allclose(yalg, yobs)


def test_direction_isolator():
    delta = np.array([1, 2, 3, 4, 5, 6])
    j = 2
    experimental = direction_isolator(delta, j)
    theoretical = np.array([0, 0, 3, 0, 0, 0])
    assert np.array_equal(theoretical, experimental)


def test_derivative():
    x = np.array([10, 20, 30, 40, 50, 60])
    delta = np.array([1, 2, 3, 4, 5, 6])
    dt = 1
    params = None
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(core).propagate(np.array([11, 20, 30, 40, 50, 60]), params).thenReturn(np.array([11, 20, 30, 40, 50, 60]))
        when(core).propagate(np.array([9, 20, 30, 40, 50, 60]), params).thenReturn(np.array([9, 20, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 22, 30, 40, 50, 60]), params).thenReturn(np.array([10, 22, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 18, 30, 40, 50, 60]), params).thenReturn(np.array([10, 18, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 33, 40, 50, 60]), params).thenReturn(np.array([10, 20, 33, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 27, 40, 50, 60]), params).thenReturn(np.array([10, 20, 27, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 44, 50, 60]), params).thenReturn(np.array([10, 20, 30, 44, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 36, 50, 60]), params).thenReturn(np.array([10, 20, 30, 36, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 55, 60]), params).thenReturn(np.array([10, 20, 30, 40, 55, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 45, 60]), params).thenReturn(np.array([10, 20, 30, 40, 45, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 50, 66]), params).thenReturn(np.array([10, 20, 30, 40, 50, 66]))
        when(core).propagate(np.array([10, 20, 30, 40, 50, 54]), params).thenReturn(np.array([10, 20, 30, 40, 50, 54]))
        when(core).y(np.array([11, 20, 30, 40, 50, 60]), params).thenReturn(np.array([2, 4]))
        when(core).y(np.array([9, 20, 30, 40, 50, 60]), params).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 22, 30, 40, 50, 60]), params).thenReturn(np.array([12, 16]))
        when(core).y(np.array([10, 18, 30, 40, 50, 60]), params).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 33, 40, 50, 60]), params).thenReturn(np.array([30, 36]))
        when(core).y(np.array([10, 20, 27, 40, 50, 60]), params).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 44, 50, 60]), params).thenReturn(np.array([56, 64]))
        when(core).y(np.array([10, 20, 30, 36, 50, 60]), params).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 40, 55, 60]), params).thenReturn(np.array([90, 100]))
        when(core).y(np.array([10, 20, 30, 40, 45, 60]), params).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 40, 50, 66]), params).thenReturn(np.array([132, 144]))
        when(core).y(np.array([10, 20, 30, 40, 50, 54]), params).thenReturn(np.array([0, 0]))
        experimental = derivative(x, delta, params, params)
        theoretical = np.array(([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]))
        assert np.array_equal(theoretical, experimental)


@pytest.mark.parametrize("delta_x, rtol, vtol, expected", [(np.array([1, 2, 3, 4, 5, 6]), 10, 10, True),
                                                           (np.array([1, 2, 3, 4, 5, 6]), 1, 1, False)])
def test_stopping_criteria(delta_x, rtol, vtol, expected):
    result = stopping_criteria(delta_x, rtol=rtol,  vtol=vtol)
    assert result == expected


def test_diagonal_form():
    a = np.array([[5, 2, -1, 0, 0],
                  [1, 4, 2, -1, 0],
                  [0, 1, 3, 2, -1],
                  [0, 0, 1, 2, 2],
                  [0, 0, 0, 1, 1]])
    b = np.array([0, 1, 2, 2, 3])
    ab = diagonal_form(a, upper=2, lower=1)
    expected = np.array([[0, 0, -1, -1, -1],
                         [0, 2, 2, 2, 2],
                         [5, 4, 3, 2, 1],
                         [1, 1, 1, 1, 0]])
    assert np.allclose(ab, expected)


def test_get_delta_x():
    a = np.array([[5, 2, -1, 0, 0],
                  [1, 4, 2, -1, 0],
                  [0, 1, 3, 2, -1],
                  [0, 0, 1, 2, 2],
                  [0, 0, 0, 1, 1]])
    b = np.array([0, 1, 2, 2, 3])
    ab = diagonal_form(a, upper=2, lower=1)
    x = solve_banded((1, 2), ab, b)
    residual = a @ x - b
    assert np.allclose(residual, np.zeros((6, 1)))
