import numpy as np
from Ffun import f
import pytest


@pytest.mark.parametrize("rr, expected", [(np.array([100, 100, 0]), np.array([45, 0])),
                                          (np.array([100, 0, 0]), np.array([0, 0])),
                                          (np.array([0, 100, 0]), np.array([90, 0])),
                                          (np.array([0, 0, -100]), np.array([0, -90]))])
def test_prediction_function(rr, expected):
    actual = f(rr, None)
    assert np.array_equal(actual, expected)
