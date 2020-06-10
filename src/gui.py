import numpy as np
from src.core import milani
from src.enums import Perturbations, Frames
from src.dto import LsqParams, Observation, PropParams
from src.util import build_j2
from src.interface.cleaning import convert_obs_params_from_lla_to_eci, verify_locational_units
from astropy.time import Time
from astropy import units as u
from src.observation_function import y
from src.state_propagator import state_propagate

x = np.array([66666, 0, 0, 0, -2.6551, 1])
xoffset = np.array([100, 50, 10, .01, .01, .03])

epoch_i = Time("2018-08-17 12:05:50", scale="tdb")
epoch_i.format = "jd"
epoch_f = epoch_i + 112 * u.day
dt = (epoch_f - epoch_i).value
obs_position = [29.2108 * u.deg, 81.0228 * u.deg, 3.9624 * u.m]     #Daytona Beach, except 13 feet above sea level (6378 km)
obs_params = Observation(obs_position, Frames.LLA, epoch_f)
obs_params = verify_locational_units(obs_params)
obs_params = convert_obs_params_from_lla_to_eci(obs_params)

prop_params = PropParams(dt, epoch_f)
prop_params.add_perturbation(Perturbations.J2, build_j2())

xobs = state_propagate(x + xoffset, prop_params)
yobs = y(xobs, obs_params)

xout = milani(x, yobs, LsqParams(), obs_params, prop_params)

print("The outcome of our algorithm is \nposition: ", xout[0:3], "\nvelocity: ", xout[3:6])
print("\nCompared to the original \nposition: ", x[0:3], "\nvelocity:", x[3:6])
print("\nDifference in observational values of x and xout")
print(yobs - y(state_propagate(xout, prop_params), obs_params))


