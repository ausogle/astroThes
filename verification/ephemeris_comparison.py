import numpy as np
from astropy.time import Time
import astropy.units as u
from verification.util import build_y_observations, build_epochs, build_eci_observations, build_icrs_observations
from src.interface.tle import tle_to_state
from src.enums import Frames
from src.interface.local_angles import get_local_angles_via_state_propagation


tle_string = """
2020-040A
1 45807U 20040A   20175.61197145 -.00000151  00000-0 -10000-3 0  9997
2 45087  28.4194 294.5058 7294772 179.9989 70.3936 2.28021460    11
"""
# 1 45807U 20040A   20175.61197145 -.00000151 00000-0 -16612-3 0  9998

x, params = tle_to_state(tle_string)

obs_pos = [29.218103 * u.deg, -81.031723 * u.deg, 0 * u.km]
epoch = Time("2020-07-03T05:00:00.000", format="isot", scale="utc")
epochs = build_epochs(epoch, 15 * u.min, 4 * 4)
y_obs = build_y_observations(x, params, obs_pos, Frames.LLA, epochs)
eci_obs = build_eci_observations(x, params, obs_pos, Frames.LLA, epochs)
icrs_obs = build_eci_observations(x, params, obs_pos, Frames.LLA, epochs)


print("Custom RA DEC")
for obs in y_obs:
    print(obs.obs_values)

print("Astropy ECI RA DEC")
for obs in eci_obs:
    print(obs)

print("Astropy ICRS RA DEC")
for obs in icrs_obs:
    print(obs)

print("Local Angles")
n = len(epochs)
locals = get_local_angles_via_state_propagation(x, params, epoch, epochs[n-1], n-2, obs_pos, Frames.LLA)
for local in locals:
    print(local)

