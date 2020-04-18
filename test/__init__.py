import mockito
import numpy as np


def xcompare(a, b):
    if isinstance(a, mockito.matchers.Matcher):
        return a.matches(b)
    return np.array_equal(a, b)
