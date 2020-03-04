import math
import numpy as np
from numpy import linalg as la


def f(rr):
    alpha = math.atan2(rr[2], rr[1]) * 180/math.pi
    dec = 90 - math.acos(rr[3]/la.norm(rr))
    return np.array([alpha, dec])
