from propagator import propagate
import numpy as np
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation
from astropy import units as u
from astropy.time import Time


def test_propagate():
    x = np.array([66666, 0, 0, 0, 2.415, 0])
    dt = 1
    params = True
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time("2018-08-17 12:05:50", scale="tbd")
    # this needs to come out of params
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    # sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=J2_perturbation, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    # output = np.concatenate([sat_f.r.value, sat_f.v.value])
    # return output