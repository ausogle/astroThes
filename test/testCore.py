import core
from core import __direction_isolator, derivative
import Ffun
import mockito
from mockito import when, patch
import propagator
import numpy as np


def test_direction_isolator():
    delta = np.array([1, 2, 3, 4, 5, 6])
    j = 2
    experimental = __direction_isolator(delta, j)
    theoretical = np.array([0, 0, 3, 0, 0, 0])
    assert xcompare(theoretical, experimental)


def test_derivative():
    x = np.array([10, 20, 30, 40, 50, 60])
    delta = np.array([1, 2, 3, 4, 5, 6])
    dt = 1
    params = True
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(core).propagate(np.array([11, 20, 30, 40, 50, 60]), dt, params).thenReturn(np.array([11, 20, 30, 40, 50, 60]))
        when(core).propagate(np.array([9, 20, 30, 40, 50, 60]), dt, params).thenReturn(np.array([9, 20, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 22, 30, 40, 50, 60]), dt, params).thenReturn(np.array([10, 22, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 18, 30, 40, 50, 60]), dt, params).thenReturn(np.array([10, 18, 30, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 33, 40, 50, 60]), dt, params).thenReturn(np.array([10, 20, 33, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 27, 40, 50, 60]), dt, params).thenReturn(np.array([10, 20, 27, 40, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 44, 50, 60]), dt, params).thenReturn(np.array([10, 20, 30, 44, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 36, 50, 60]), dt, params).thenReturn(np.array([10, 20, 30, 36, 50, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 55, 60]), dt, params).thenReturn(np.array([10, 20, 30, 40, 55, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 45, 60]), dt, params).thenReturn(np.array([10, 20, 30, 40, 45, 60]))
        when(core).propagate(np.array([10, 20, 30, 40, 50, 66]), dt, params).thenReturn(np.array([10, 20, 30, 40, 50, 66]))
        when(core).propagate(np.array([10, 20, 30, 40, 50, 54]), dt, params).thenReturn(np.array([10, 20, 30, 40, 50, 54]))
        when(core).f(np.array([11, 20, 30, 40, 50, 60])).thenReturn(np.array([2, 4]))
        when(core).f(np.array([9, 20, 30, 40, 50, 60])).thenReturn(np.array([0, 0]))
        when(core).f(np.array([10, 22, 30, 40, 50, 60])).thenReturn(np.array([12, 16]))
        when(core).f(np.array([10, 18, 30, 40, 50, 60])).thenReturn(np.array([0, 0]))
        when(core).f(np.array([10, 20, 33, 40, 50, 60])).thenReturn(np.array([30, 36]))
        when(core).f(np.array([10, 20, 27, 40, 50, 60])).thenReturn(np.array([0, 0]))
        when(core).f(np.array([10, 20, 30, 44, 50, 60])).thenReturn(np.array([56, 64]))
        when(core).f(np.array([10, 20, 30, 36, 50, 60])).thenReturn(np.array([0, 0]))
        when(core).f(np.array([10, 20, 30, 40, 55, 60])).thenReturn(np.array([90, 100]))
        when(core).f(np.array([10, 20, 30, 40, 45, 60])).thenReturn(np.array([0, 0]))
        when(core).f(np.array([10, 20, 30, 40, 50, 66])).thenReturn(np.array([132, 144]))
        when(core).f(np.array([10, 20, 30, 40, 50, 54])).thenReturn(np.array([0, 0]))
        experimental = derivative(x, delta, dt, params)
        theoretical = np.array(([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]))
        assert xcompare(theoretical, experimental)


def xcompare(a, b):
    if isinstance(a, mockito.matchers.Matcher):
        return a.matches(b)

    return np.array_equal(a, b)


