import numpy as np
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag, J3_perturbation
from poliastro.core.perturbations import third_body as three_body
from astropy import units as u
from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris


def propagate(x, params):
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=params.epoch)
    sat_f = sat_i.propagate(params.dt * u.s, method=cowell, ad=J2_perturbation, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    output = np.concatenate([sat_f.r.value, sat_f.v.value])
    return output


def perturbation_accel(params, t0, state, k):
    functions = []
    if "J2" in params.perturbations:
        perturbation = params.perturbations.get("J2")
        functions.append(J2_perturbation(t0, state, k, perturbation.J2, perturbation.R))
    if "J3" in params.perturbations:
        perturbation = params.perturbations.get("J3")
        functions.append(J3_perturbation(t0, state, k, perturbation.J3, perturbation.R))
    if "Drag" in params.perturbations:
        perturbation = params.perturbations.get("Drag")
        functions.append(atmospheric_drag(t0, state, k, perturbation.R, perturbation.C_D,
                                          perturbation.A, perturbation.m, perturbation.H0, perturbation.rh0))
    if "Moon" in params.perturbations:
        perturbation = params.perturbations.get("Moon")
        functions.append(three_body(t0, state, k, perturbation.k_third, perturbation.third_body))


def a_d(t0, state, k, J2, J3, R, C_D, A, m, H0, rho0, k_third, third_body):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag(t0, state, k, R, C_D, A, m, H0, rho0) \
        + J3_perturbation(t0, state, k, J3, R) + three_body(t0, state, k, k_third, third_body)


