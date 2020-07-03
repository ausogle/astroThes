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
from src.interface.local_angles import get_local_angles_via_state_propagation, local_angles


tle_string = """
2020-040A
1 45807U 20040A   20175.61197145 -.00000151  00000-0 -10000-3 0  9998
2 45087  28.4194 294.5058 7294772 179.9989 70.3936 2.28021460    11
"""
# 1 45807U 20040A   20175.61197145 -.00000151 00000-0 -16612-3 0  9998

x, params = tle_to_state(tle_string)
epoch = Time("2020-06-23T00:00:00.000", format="isot", scale="utc")
params.epoch = epoch
obs_pos = [29.218103 * u.deg, -81.031723 * u.deg, 0 * u.km]

epochs = build_epochs(epoch, 15 * u.min, 4 * 24)
observations = build_observations(x, params, obs_pos, Frames.LLA, epochs)

for obs in observations:
    print(obs.obs_values)

# n = len(epochs)
# locals = get_local_angles_via_state_propagation(x, params, epoch, epochs[n-1], n-2, obs_pos, Frames.LLA)
# for local in locals:
#     print(local)

