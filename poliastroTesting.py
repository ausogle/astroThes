import numpy as np
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth, Moon
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag, J3_perturbation
from poliastro.core.perturbations import third_body as three_body
from poliastro.core.util import jit
from poliastro.ephem import build_ephem_interpolant

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris

x = [66666, 0, 0, 0, 2.451, 0]
dt = 1
r = x[0:3] * u.km
v = x[3:6] * u.km / u.s
# epoch = Time("2018-08-17 12:05:50", scale="tdb")
epoch = Time(2454283.0, format="jd", scale="tdb")
solar_system_ephemeris.set("de432s")
body_moon = build_ephem_interpolant(Moon, 28 * u.day, (epoch.value * u.day, epoch.value * u.day + 60 * u.day), rtol=1e-2)


def a_d(t0, state, k, J2, J3, R, C_D, A, m, H0, rho0, k_third, third_body):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag(t0, state, k, R, C_D, A, m, H0, rho0) \
           + J3_perturbation(t0, state, k, J3, R) + three_body(t0, state, k, k_third, third_body)


C_D = 1
A = 10
m = 1000
H0 = 100
rho0 = 1000

sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, J2=Earth.J2.value, J3=Earth.J3.value, R=Earth.R.to(u.km).value,
                        C_D=C_D, A=A, m=m, H0=H0, rho0=rho0, k_third=Moon.k.to(u.km ** 3 / u.s ** 2).value, third_body=body_moon)

output = np.concatenate([sat_f.r.value, sat_f.v.value])
print(output)

sat_moon = sat_i.propagate(dt * u.s, method=cowell, ad=a_2, k_third=Moon.k.to(u.km ** 3 / u.s ** 2).value, third_body=body_moon)
print(sat_moon.r)
print(sat_moon.v)
