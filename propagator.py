import numpy as np
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag, J3_perturbation
from poliastro.core.perturbations import third_body as three_body
from astropy import units as u


# from poliastro.ephem import build_ephem_interpolant
# from astropy.coordinates import solar_system_ephemeris
# Need to make the third body values here


def propagate(x, params):
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=params.epoch)
    sat_f = sat_i.propagate(params.dt * u.s, method=cowell, ad=a_d, perturbations=params.perturbations)
    output = np.concatenate([sat_f.r.value, sat_f.v.value])
    return output


def a_d(t0, state, k, perturbations):
    fun = []
    if "J2" in perturbations:
        perturbation = perturbations.get("J2")
        fun.append(J2_perturbation(t0, state, k, perturbation.J2, perturbation.R))
    if "Drag" in perturbations:
        perturbation = perturbations.get("Drag")
        fun.append(atmospheric_drag(t0, state, k, perturbation.R, perturbation.C_D, perturbation.A, perturbation.m,
                                    perturbation.H0, perturbation.rho0))
    if "J3" in perturbations:
        perturbation = perturbations.get("J3")
        fun.append(J3_perturbation(t0, state, k, perturbation.J3, perturbation.R))
    # if "Moon" in perturbations:
    #     perturbation = perturbations.get("Moon")
    #     fun.append(three_body(t0, state, k, perturbation.k_third, perturbation.third_body))
    # Need to create these objects in this method
    # epoch = Time(2454283.0, format="jd", scale="tdb")
    # solar_system_ephemeris.set("de432s")
    # body_moon = build_ephem_interpolant(Moon, 28 * u.day, (epoch.value * u.day, epoch.value * u.day + 60 * u.day),
    #                                     rtol=1e-2)
    # moon = ThirdBody(Moon.k.to(u.km ** 3 / u.s ** 2).value, body_moon)
    # prop_params.add_perturbation("Moon", moon)

    # To add additional, or improvements upon existing perturbations, everything must be included in this function.

    def summation(lost):
        if len(lost) == 0:
            return None
        output = lost[0]
        for i in range(1, len(lost)):
            output += lost[i]
        return output

    return summation(fun)
