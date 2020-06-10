import numpy as np
import mockito
from mockito import when, patch
from test import xcompare
from src import covariance_propagator
from src.covariance_propagator import *


def test_cov_propagate():
    x = np.array([1, 1, 1, 1, 1, 1])
    epoch_t = None
    prop_params = None
    dr = .1
    dv = .005
    delta = np.array([dr, dr, dr, dv, dv, dv])
    p_i = np.ones((6, 6))

    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(covariance_propagator).dx_dx0(x, epoch_t, prop_params, delta).thenReturn(np.eye(6))
        p_t = cov_propagate(x, epoch_t, prop_params, p_i)
    assert np.array_equal(p_i, p_t)


def test_dx_dx0():
    x = np.array([10, 20, 30, 40, 50, 60])
    delta = np.array([1, 2, 3, 4, 5, 6])
    epoch_t = None
    params = None

    expected = np.array([[1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12],
                         [13, 14, 15, 16, 17, 18],
                         [19, 20, 21, 22, 23, 24],
                         [25, 26, 27, 28, 29, 30],
                         [31, 32, 33, 34, 35, 36]])

    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(covariance_propagator).state_propagate(np.array([11, 20, 30, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([2, 28, 78, 152, 250, 372]))
        when(covariance_propagator).state_propagate(np.array([9, 20, 30, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        when(covariance_propagator).state_propagate(np.array([10, 22, 30, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([4, 32, 84, 160, 260, 384]))
        when(covariance_propagator).state_propagate(np.array([10, 18, 30, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 33, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([6, 36, 90, 168, 270, 396]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 27, 40, 50, 60]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 44, 50, 60]), epoch_t, params).thenReturn(
            np.array([8, 40, 96, 176, 280, 408]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 36, 50, 60]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 40, 55, 60]), epoch_t, params).thenReturn(
            np.array([10, 44, 102, 184, 290, 420]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 40, 45, 60]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 40, 50, 66]), epoch_t, params).thenReturn(
            np.array([12, 48, 108, 192, 300, 432]))
        when(covariance_propagator).state_propagate(np.array([10, 20, 30, 40, 50, 54]), epoch_t, params).thenReturn(
            np.array([0, 0, 0, 0, 0, 0]))
        result = dx_dx0(x, epoch_t, params, delta)
    assert np.array_equal(expected,  result)
