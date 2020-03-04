import numpy as np
from core import core


x = np.array([66666, 0, 0, 0, -1.4551, 0])
xoffset = np.array([100, 0, 0, 0, 0, 0])
dt = 100
params = "Hi"

xout = core(x, xoffset, dt, params)

print(xout)
