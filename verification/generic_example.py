from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from verification.util import generate_earth_surface, get_satellite_position_over_time
from src.core import milani
from src.dto import PropParams, Observation
from src.enums import Frames, Angles, Perturbations
from src.state_propagator import state_propagate
from src.util import build_j2, build_j3
from src.observation_function import y
from astropy.time import Time
import astropy.units as u

x_guess = np.array([5975.2904, 2568.6400, 3120.5845, 3.983846, -2.071159, -5.917095])
x_expected = np.array([5748.6011, 2679.7287, 3443.0073, 4.330462, -1.922862, -5.726564])
epoch_i = Time(2449746.610150, format="jd", scale="utc")

obs_pos = [21.571831046 * u.deg, -158.2761 * u.deg, 3.9624 * u.km]     #Kaena Point, except 13 feet above sea level
obs_sigmas = np.array([.0925, .0224, .0139])
epoch_obs1 = Time('1995-1-29T02:38:37', format='isot', scale='utc')
obs1 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 0 * u.s, np.array([2047.502, 60.4991, 16.1932]), Angles.Local, obs_sigmas)
obs2 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 12 * u.s, np.array([1984.677, 62.1435, 17.2761]), Angles.Local, obs_sigmas)
obs3 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 25 * u.s, np.array([1918.489, 64.0566, 18.5515]), Angles.Local, obs_sigmas)
obs4 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 37 * u.s, np.array([1859.320, 65.8882, 19.7261]), Angles.Local, obs_sigmas)
obs5 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 49 * u.s, np.array([1802.186, 67.9320, 20.9351]), Angles.Local, obs_sigmas)
obs6 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 61 * u.s, np.array([1747.290, 70.1187, 22.1319]), Angles.Local, obs_sigmas)
obs7 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 73 * u.s, np.array([1694.891, 72.5159, 23.3891]), Angles.Local, obs_sigmas)
obs8 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 86 * u.s, np.array([1641.201, 75.3066, 24.7484]), Angles.Local, obs_sigmas)
obs9 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 99 * u.s, np.array([1594.770, 78.1000, 25.9799]), Angles.Local, obs_sigmas)
obs10 = Observation(obs_pos, Frames.LLA, epoch_obs1 + 111 * u.s, np.array([1551.640, 81.1197, 27.1896]), Angles.Local, obs_sigmas)
observations = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]

prop_params = PropParams(epoch_i)
prop_params.add_perturbation(Perturbations.J2, build_j2())
prop_params.add_perturbation(Perturbations.J3, build_j3())

print(y(state_propagate(x_expected, obs1.epoch, prop_params), obs1))
print(y(state_propagate(x_expected, obs2.epoch, prop_params), obs2))
print(y(state_propagate(x_expected, obs3.epoch, prop_params), obs3))
print(y(state_propagate(x_expected, obs4.epoch, prop_params), obs4))
print(y(state_propagate(x_expected, obs5.epoch, prop_params), obs5))
print(y(state_propagate(x_expected, obs6.epoch, prop_params), obs6))
print(y(state_propagate(x_expected, obs7.epoch, prop_params), obs7))
print(y(state_propagate(x_expected, obs8.epoch, prop_params), obs8))
print(y(state_propagate(x_expected, obs9.epoch, prop_params), obs9))
print(y(state_propagate(x_expected, obs10.epoch, prop_params), obs10))

x_alg, p = milani(x_guess, observations, prop_params)
print("algorithm output")
print(x_alg)
print("stdev squared")
print(np.diag(p))
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
