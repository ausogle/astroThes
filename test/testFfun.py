import numpy as np
from Ffun import f


def test_f_1():
    rr = np.array([100, 100, 0])
    theoretical = np.array([45, 0])
    experimental = f(rr, None)
    comparison = theoretical == experimental
    assert comparison.all()


def test_f_2():
    rr = np.array([100, 0, 0])
    theoretical = np.array([0, 0])
    experimental = f(rr, None)
    comparison = theoretical == experimental
    assert comparison.all()


def test_f_3():
    rr = np.array([0, 100, 0])
    theoretical = np.array([90, 0])
    experimental = f(rr, None)
    comparison = theoretical == experimental
    assert comparison.all()


def test_f_4():
    rr = np.array([0, 0, -100])
    theoretical = np.array([0, -90])
    experimental = f(rr, None)
    comparison = theoretical == experimental
    assert comparison.all()
