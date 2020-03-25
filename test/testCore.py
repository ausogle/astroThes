from core import __direction_isolator
import numpy as np


def test_direction_isolator():
    delta = np.array([1, 2, 3, 4, 5, 6])
    j = 2
    experimental = __direction_isolator(delta, j)
    theoretical = np.array([0, 0, 3, 0, 0, 0])
    comparison = theoretical == experimental
    assert comparison.all()
