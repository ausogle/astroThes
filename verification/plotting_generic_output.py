from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from verification.util import generate_earth_surface, get_satellite_position_over_time
from src.core import milani
from src.dto import PropParams, ObsParams
from src.enums import Frames
from astropy.time import Time
from astropy import units as u
from verification.util import get_period

r = [66666, 0, 0]
v = [0, -2.644, .01]
x = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
period = get_period(x)
dt = period / 100
tf = period / 2
epoch = Time(2454283.0, format="jd", scale="tdb")

x_offset = np.array([100, 50, 10, .01, .01, .03])
obs_loc = ["lat", "lon", "alt"]
obs_frame = Frames.ECEF.value
obs_params = ObsParams(obs_loc, obs_frame, epoch)
epoch_f = epoch + tf * u.s
prop_params = PropParams(dt, epoch_f)
x_alg = milani(x, x_offset, obs_params, prop_params)

r_init = get_satellite_position_over_time(x, epoch, tf, dt)
r_offset = get_satellite_position_over_time(x + x_offset, epoch, tf, dt)
r_alg = get_satellite_position_over_time(x_alg, epoch, tf, dt)

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = generate_earth_surface()
ax.plot3D(r_init[:, 0], r_init[:, 1], r_init[:, 2], color='red', label='initial')
ax.plot3D(r_offset[:, 0], r_offset[:, 1], r_offset[:, 2], color='yellow', label='offset')
ax.plot3D(r_alg[:, 0], r_alg[:, 1], r_alg[:, 2], color='green', label='algorithm')
ax.plot_surface(x, y, z, color='b')

Re = 6378
dim = 6378 * 15
ax.set_xlim([-dim, 2*dim])
ax.set_ylim([-dim, 2*dim])
ax.set_zlim([-dim, 2*dim])
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.legend()
plt.show()
