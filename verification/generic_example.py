from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from verification.util import generate_earth_surface, get_satellite_position_over_time, build_observations, build_epochs
from src.core import milani
from src.dto import PropParams
from src.enums import Frames
from astropy.time import Time
import astropy.units as u
from verification.util import get_period

r = [5748.6001, 2679, 3443]
v = [4.33, -1.922, -5.726]
x = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
x_offset = np.array([500, 100, 100, .2, .1, .1])
x_true = x + x_offset
period = get_period(x)
dt = period / 100
tf = period * 2
epoch = Time(2454283.0, format="jd", scale="tdb")

obs_pos = [29.2108 * u.deg, 81.0228 * u.deg, 3.9624 * u.km]     #Daytona Beach, except 13 feet above sea level
prop_params = PropParams(epoch)
step = period/32 * u.s
epochs = build_epochs(epoch, step, 5)
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

# r_init = get_satellite_position_over_time(x, epoch_obs, tf, dt)
# r_offset = get_satellite_position_over_time(x + x_offset, epoch_obs, tf, dt)
# r_alg = get_satellite_position_over_time(x_alg, epoch_obs, tf, dt)
#
# n = r_alg.shape[0]-1
# print(r_offset[n, 0])
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# x, y, z = generate_earth_surface()
# ax.plot3D(r_init[:, 0], r_init[:, 1], r_init[:, 2], color='red', label='initial')
# ax.plot3D(r_offset[:, 0], r_offset[:, 1], r_offset[:, 2], color='blue', label='offset')
# ax.plot3D(r_alg[:, 0], r_alg[:, 1], r_alg[:, 2], color='green', label='algorithm')
# ax.plot3D([r_offset[n, 0]], [r_offset[n, 1]], [r_offset[n, 2]], color='blue', label='Final Location', marker='o')
# ax.plot_surface(x, y, z, color='b')
#
# Re = 6378
# dim = 6378 * 15
# ax.set_xlim([-dim, dim])
# ax.set_ylim([-dim, dim])
# ax.set_zlim([-dim, dim])
# ax.set_xlabel('x [km]')
# ax.set_ylabel('y [km]')
# ax.set_zlabel('z [km]')
# ax.legend()
# plt.show()
#