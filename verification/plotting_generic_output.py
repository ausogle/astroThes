from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from verification.util import generate_earth_surface, get_satellite_position_over_time
from src.core import milani
from src.dto import PropParams, ObsParams
from src.enums import Frames
from src.propagator import propagate
from src.ffun import f
from src.util import convert_obs_params_from_lla_to_ecef
from astropy.time import Time
import astropy.units as u
from verification.util import get_period

r = [66666, 0, 0]
v = [0, -2.144, 1]
x = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
period = get_period(x)
dt = period / 100
tf = period / 2
epoch_obs = Time(2454283.0, format="jd", scale="tdb")
epoch_i = epoch_obs - tf * u.s

x_offset = np.array([100, 50, 10, .01, .01, .03])
obs_pos = [29.2108, 81.0228, 3.9624]     #Daytona Beach, except 13 feet above sea level (6378 km)
obs_params = ObsParams(obs_pos, Frames.LLA, epoch_obs)
obs_params = convert_obs_params_from_lla_to_ecef(obs_params)
prop_params = PropParams(tf, epoch_i)
yobs = f(propagate(x+x_offset, prop_params), obs_params)
x_alg = milani(x, yobs, obs_params, prop_params)

r_init = get_satellite_position_over_time(x, epoch_obs, tf, dt)
r_offset = get_satellite_position_over_time(x + x_offset, epoch_obs, tf, dt)
r_alg = get_satellite_position_over_time(x_alg, epoch_obs, tf, dt)

n = r_alg.shape[0]-1
print(r_offset[n, 0])

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = generate_earth_surface()
ax.plot3D(r_init[:, 0], r_init[:, 1], r_init[:, 2], color='red', label='initial')
ax.plot3D(r_offset[:, 0], r_offset[:, 1], r_offset[:, 2], color='blue', label='offset')
ax.plot3D(r_alg[:, 0], r_alg[:, 1], r_alg[:, 2], color='green', label='algorithm')
ax.plot3D([r_offset[n, 0]], [r_offset[n, 1]], [r_offset[n, 2]], color='blue', label='Final Location', marker='o')
ax.plot_surface(x, y, z, color='b')

Re = 6378
dim = 6378 * 15
ax.set_xlim([-dim, dim])
ax.set_ylim([-dim, dim])
ax.set_zlim([-dim, dim])
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.legend()
plt.show()
