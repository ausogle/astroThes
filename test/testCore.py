from core import __direction_isolator, __derivative
import Ffun
import mockito
from mockito import when, mock, args, patch
import propagator
import numpy as np
from unittest.mock import MagicMock

#
# def test_direction_isolator():
#     delta = np.array([1, 2, 3, 4, 5, 6])
#     j = 2
#     experimental = __direction_isolator(delta, j)
#     theoretical = np.array([0, 0, 3, 0, 0, 0])
#     comparison = theoretical == experimental
#     assert comparison.all()
#
#
# def test_derivative():
#     x = np.array([10, 20, 30, 40, 50, 60])
#     delta = np.array([1, 2, 3, 4, 5, 6])
#     dt = 1
#
#     mock = propagator
#     mock.propagate = MagicMock(name='method')
#     mock.side_effect = [3, 2, 1]
#
#     theoretical = np.array(([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]))
#     experimental = __derivative(x, delta, dt, params="nothing")
#     comparison = theoretical == experimental
#     assert comparison.all()


def fun(rr):
    return Ffun.fuck_me(rr)


def test_fun():
    sample = np.array([100, 100, 1])
    output = np.array([100, 1])
    with patch(mockito.invocation.MatchingInvocation.compare, xcompare):
        when(Ffun).fuck_me(sample).thenReturn(output)
        result = fun(sample)
        assert xcompare(result, output)


def xcompare(a, b):
    if isinstance(a, mockito.matchers.Matcher):
        return a.matches(b)

    return np.array_equal(a, b)
