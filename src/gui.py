import numpy as np
from src.core import milani
from src.enums import Perturbations, Frames
from src.dto import ObsParams, PropParams, J2
from poliastro.bodies import Earth
from astropy.time import Time
from astropy import units as u
from ffun import f

x = np.array([66666, 0, 0, 0, -1.4551, 0])
xoffset = np.array([1000, 300, 150, .01, .01, 0])

epoch_i = Time("2018-08-17 12:05:50", scale="tdb")
epoch_i.format = "jd"
epoch_f = epoch_i + 1 * u.day
dt = (epoch_f - epoch_i).value
obs_loc = ["lat", "lon", "alt"]
obs_frame = Frames.ECEF.value
obs_params = ObsParams(obs_loc, obs_frame, epoch_i)

J2 = J2(Earth.J2.value, Earth.R.to(u.km).value)
prop_params = PropParams(dt, epoch_f)
prop_params.add_perturbation(Perturbations.J2.value, J2)

xout = milani(x, xoffset, obs_params, prop_params)

print("The outcome of our algorithm is \nposition: ", xout[0:3], "\nvelocity: ", xout[3:6])
print("\nCompared to the original \nposition: ", x[0:3], "\nvelocity:", x[3:6])
print("\nDifference in observational values of x and xout")
print(f(x, obs_params), f(xout, obs_params))
