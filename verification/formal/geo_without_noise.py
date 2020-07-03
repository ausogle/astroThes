import numpy as np
from verification.util import build_observations, build_epochs
from src.core import milani
from src.dto import PropParams
from src.enums import Frames
from astropy.time import Time
import astropy.units as u
from verification.util import get_period

r = [-27828.9136, -31685.0220, 3.5110]
v = [2.3098, -2.0286, -.0019]
x = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
x_offset = np.array([5000, 1000, 1000, .2, .2, .01])
x_true = x + x_offset
period = get_period(x)
dt = period / 100
tf = period * 2
epoch = Time(2449746.610150, format="jd", scale="utc")

obs_pos = [21.57 * u.deg, -158.27 * u.deg, .3002 * u.km] # Kaena Point, HI
prop_params = PropParams(epoch)
step = period/32 * u.s
epochs = build_epochs(epoch, step, 10)
observations = build_observations(x_true, prop_params, obs_pos, Frames.LLA, epochs)
output = milani(x, observations, prop_params)
x_alg = output.x_out
p = output.p

print("x alg")
print(x_alg)
print("x true")
print(x_true)

print("State residual")
print(x_true - x_alg)
print("Uncertainty")
print(np.diag(p))
print("initial offset")
print(x_offset)