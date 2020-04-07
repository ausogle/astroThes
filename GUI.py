import numpy as np
from core import milano
from util import ObsParams, PropParams, J2, J3, Drag, ThirdBody
from poliastro.bodies import Earth, Moon
from astropy.time import Time
from astropy import units as u

x = np.array([66666, 0, 0, 0, -1.4551, 0])
xoffset = np.array([100, 0, 0, 0, 0, 0])

epoch_i = Time("2018-08-17 12:05:50", scale="tdb")
epoch_i.format = "jd"
epoch_f = epoch_i + 1 * u.day
dt = (epoch_f - epoch_i).value
obs_loc = ["lat", "long", "alt"]
obs_frame = "ECEF"
obs_params = ObsParams(obs_loc, obs_frame, epoch_f)

J2 = J2(Earth.J2.value, Earth.R.to(u.km).value)
prop_params = PropParams(dt, epoch_f)
prop_params.add_perturbation("J2", J2)

print(prop_params.dt)

# xout = milano(x, xoffset, dt, obs_params, prop_params)
# print(xout)
