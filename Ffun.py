import math
import numpy as np
from numpy import linalg as la


def f(rr, obs_params):
    alpha = math.atan2(rr[1], rr[0]) * 180/math.pi
    dec = 90 - math.acos(rr[2]/la.norm(rr))*180/math.pi
    return np.array([alpha, dec])

