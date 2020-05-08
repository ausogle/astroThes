import numpy as np
from src.enums import Perturbations
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag, J3_perturbation, radiation_pressure
from astropy import units as u
from src.dto import PropParams
from typing import Dict


# from poliastro.ephem import build_ephem_interpolant
# from astropy.coordinates import solar_system_ephemeris
# Need to make the third body values here


def propagate(x: np.ndarray, params: PropParams) -> np.ndarray:
    """
    Propagates the state vector from moment of description to moment of observation in time another using the poliasto
    library. Allows for custom perturbations.
    :param x: State vector at time of original description
    :param params: object which serves as catch all for relevant info. Includes dt or amount of time between initial
    description to moment of observation, epoch of initial description, and perturbations to be included.
    :return: Returns state vector at moment of observation
    """
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=params.epoch)
    sat_f = sat_i.propagate(params.dt * u.s, method=cowell, ad=a_d, perturbations=params.perturbations)
    output = np.concatenate([sat_f.r.value, sat_f.v.value])
    return output


def a_d(t0, state, k, perturbations: Dict):
    """
    Custom perturbation function that is passed directly to poliastro to be executed in their code, hence the need for
    summation() to be included within. Current structure allows user to pick and chose which perturbations they would
    like to include, requiring that the desired perturbation objects are created, filled, and passed.
    :param t0: Required by poliastro
    :param state: Required by poliastro
    :param k: Required by poliastro (gravitational parameter-mu)
    :param perturbations: Dictionary of perturbations desired by the user. Keys correspond to the perturbations Enum
    class in Enum.py, while values correspond to objects in the dto.py class.
    :return: Returns a force that describes the impact of all desired perturbations
    """
    fun = []
    if Perturbations.J2.value in perturbations:
        perturbation = perturbations.get(Perturbations.J2.value)
        fun.append(J2_perturbation(t0, state, k, perturbation.J2, perturbation.R))
    if Perturbations.Drag.value in perturbations:
        perturbation = perturbations.get(Perturbations.Drag.value)
        fun.append(atmospheric_drag(t0, state, k, perturbation.R, perturbation.C_D, perturbation.A, perturbation.m,
                                    perturbation.H0, perturbation.rho0))
    if Perturbations.J3.value in perturbations:
        perturbation = perturbations.get(Perturbations.J3.value)
        fun.append(J3_perturbation(t0, state, k, perturbation.J3, perturbation.R))
    if Perturbations.SRP.value in perturbations:
        perturbation = perturbations.get(Perturbations.SRP.value)
        fun.append(radiation_pressure(t0, state, k, perturbation.R, perturbation.C_R, perturbation.A, perturbation.m,
                                      perturbation.Wdivc_s, perturbation.star))
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

    def summation(arr):
        if len(arr) == 0:
            return None
        output = arr[0]
        for i in range(1, len(arr)):
            output += arr[i]
        return output

    return summation(fun)
