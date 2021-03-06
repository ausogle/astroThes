from src import core
from src.core import *
import mockito
from mockito import when, patch
import pytest
import numpy as np
from test import xcompare


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
    epoch_obs = None
    observation = Observation(None, None, epoch_obs, None, None)
    params = None
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(core).state_propagate(np.array([11, 20, 30, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([11, 20, 30, 40, 50, 60]))
        when(core).state_propagate(np.array([9, 20, 30, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([9, 20, 30, 40, 50, 60]))
        when(core).state_propagate(np.array([10, 22, 30, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 22, 30, 40, 50, 60]))
        when(core).state_propagate(np.array([10, 18, 30, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 18, 30, 40, 50, 60]))
        when(core).state_propagate(np.array([10, 20, 33, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 33, 40, 50, 60]))
        when(core).state_propagate(np.array([10, 20, 27, 40, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 27, 40, 50, 60]))
        when(core).state_propagate(np.array([10, 20, 30, 44, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 44, 50, 60]))
        when(core).state_propagate(np.array([10, 20, 30, 36, 50, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 36, 50, 60]))
        when(core).state_propagate(np.array([10, 20, 30, 40, 55, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 40, 55, 60]))
        when(core).state_propagate(np.array([10, 20, 30, 40, 45, 60]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 40, 45, 60]))
        when(core).state_propagate(np.array([10, 20, 30, 40, 50, 66]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 40, 50, 66]))
        when(core).state_propagate(np.array([10, 20, 30, 40, 50, 54]), epoch_obs, params).thenReturn(np.array([10, 20, 30, 40, 50, 54]))
        when(core).y(np.array([11, 20, 30, 40, 50, 60]), observation).thenReturn(np.array([2, 4]))
        when(core).y(np.array([9, 20, 30, 40, 50, 60]), observation).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 22, 30, 40, 50, 60]), observation).thenReturn(np.array([12, 16]))
        when(core).y(np.array([10, 18, 30, 40, 50, 60]), observation).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 33, 40, 50, 60]), observation).thenReturn(np.array([30, 36]))
        when(core).y(np.array([10, 20, 27, 40, 50, 60]), observation).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 44, 50, 60]), observation).thenReturn(np.array([56, 64]))
        when(core).y(np.array([10, 20, 30, 36, 50, 60]), observation).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 40, 55, 60]), observation).thenReturn(np.array([90, 100]))
        when(core).y(np.array([10, 20, 30, 40, 45, 60]), observation).thenReturn(np.array([0, 0]))
        when(core).y(np.array([10, 20, 30, 40, 50, 66]), observation).thenReturn(np.array([132, 144]))
        when(core).y(np.array([10, 20, 30, 40, 50, 54]), observation).thenReturn(np.array([0, 0]))
        experimental = dy_dstate(x, delta, observation, params)
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
