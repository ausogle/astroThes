from astropy.time import Time
import astropy.units as u
from src.enums import Frames
from src.observation_function import y
from src.state_propagator import state_propagate
from src.dto import PropParams, Observation
from src.interface.cleaning import convert_obs_from_lla_to_eci
from src.interface.local_angles import get_local_angles_via_state_propagation
from src.interface.tle_dto import TLE

obs_pos = [29.218103 * u.deg, -81.031723 * u.deg, 0 * u.km]
tle_string = """
ISS (ZARYA)             
1 25544U 98067A   20211.19695584  .00000552  00000-0  17878-4 0  9990
2 25544  51.6419 140.5349 0000882 143.5591 191.9640 15.49512638238524
"""
tle = TLE.from_lines(tle_string)
x, epoch_i = tle.to_state()

epoch_1 = Time("2020-7-25T00:00:00.000")
epoch_2 = Time("2020-7-25T00:05:00.000")
epoch_3 = Time("2020-7-25T00:10:00.000")
epoch_4 = Time("2020-7-25T00:15:00.000")

x_1 = state_propagate(x, epoch_i, PropParams(epoch_1))
x_2 = state_propagate(x, epoch_i, PropParams(epoch_2))
x_3 = state_propagate(x, epoch_i, PropParams(epoch_3))
x_4 = state_propagate(x, epoch_i, PropParams(epoch_4))

observation_1 = convert_obs_from_lla_to_eci(Observation(obs_pos, Frames.LLA, epoch_1, None, None, None))
observation_2 = convert_obs_from_lla_to_eci(Observation(obs_pos, Frames.LLA, epoch_2, None, None, None))
observation_3 = convert_obs_from_lla_to_eci(Observation(obs_pos, Frames.LLA, epoch_3, None, None, None))
observation_4 = convert_obs_from_lla_to_eci(Observation(obs_pos, Frames.LLA, epoch_4, None, None, None))

print(y(x_1, observation_1))
print(y(x_2, observation_2))
print(y(x_3, observation_3))
print(y(x_4, observation_4))

