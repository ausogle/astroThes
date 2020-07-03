import math
import numpy as np
from astropy.time import Time
import astropy.units as u
from verification.util import get_satellite_position_over_time, build_observations, build_epochs
from src.interface.tle import tle_to_state
from src.enums import Frames
from src.dto import PropParams, Observation
from src.observation_function import y
from src.state_propagator import state_propagate
from src.frames import eci_to_ecef, lla_to_ecef
from src.interface.cleaning import convert_obs_from_lla_to_eci
from src.interface.local_angles import get_local_angles_for_state_propagation, local_angles


# tle_string = """
# 2020-040A
# 1 45807U 20040A   20175.61197145 -.00009734  00000-0 -10000-3 0  9998
# 2 45087  28.4194 294.5058 7294772 179.9989 70.3936 2.28021460    11
# """
# # 1 45807U 20040A   20175.61197145 -.00000151 00000-0 -16612-3 0  9998
#
# x, params = tle_to_state(tle_string)
# epoch = Time("2020-07-03T05:00:00.000", format="isot", scale="utc")
# params.epoch = epoch
# obs_pos = [29.218103 * u.deg, -81.031723 * u.deg, 0 * u.km]
#
# epochs = build_epochs(epoch, 15 * u.min, 4)
# observations = build_observations(x, params, obs_pos, Frames.LLA, epochs)
#
# for obs in observations:
#     print(obs.obs_values)
#
# n = len(epochs)
# locals = get_local_angles_for_state_propagation(x, params, epochs[n-1], n-2, obs_pos, Frames.LLA)
# for local in locals:
#     print(local)

x = [5748.6011, 2679.7287, 3443.0073, 4.330462, -1.922862, -5.726564]
epoch = Time(2449746.610150, format="jd", scale="utc")
epoch.format = "isot"
params = PropParams(epoch)

epoch_i = Time("1995-01-29T02:38:37.000", format="isot", scale="utc")
epoch_f = Time("1995-01-29T02:40:27.000", format="isot", scale="utc")
obs_pos = [21.57 * u.deg, -158.27 * u.deg, .3 * u.km]
# locals = get_local_angles_for_state_propagation(x, params, epoch_i, epoch_f, 8, obs_pos, Frames.LLA)
# for local in locals:
#     print(local)

obs = Observation(obs_pos, Frames.LLA, epoch_i, None, None)
obs = convert_obs_from_lla_to_eci(obs)

x_epoch = eci_to_ecef(state_propagate(x, epoch_i, params), epoch_i)
obs_epoch = lla_to_ecef(obs_pos)
rr = x_epoch - obs_epoch
print(local_angles(rr, obs_pos))
print(np.linalg.norm(rr))