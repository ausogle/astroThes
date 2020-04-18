import numpy as np
from propagator import a_d
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth, Moon
from poliastro.core.perturbations import J2_perturbation, J3_perturbation, atmospheric_drag, third_body
# from poliastro.ephem import build_ephem_interpolant
# from astropy.coordinates import solar_system_ephemeris
from astropy import units as u
from astropy.time import Time
import util


def test_propagate_with_j2j3():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")

    J2 = util.J2(Earth.J2.value, Earth.R.to(u.km).value)
    J3 = util.J3(Earth.J3.value, Earth.R.to(u.km).value)
    prop_params = util.PropParams(1, epoch)
    prop_params.add_perturbation("J2", J2)
    prop_params.add_perturbation("J3", J3)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=a_d_j2j3, J2=Earth.J2.value,
                            R=Earth.R.to(u.km).value, J3=Earth.J3.value)
    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def test_propagate_with_drag():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    C_D = 1
    A = 10
    m = 1000
    H0 = 100
    rho0 = 1000
    Drag = util.Drag(Earth.R.to(u.km).value, C_D, A, m, H0, rho0)
    prop_params = util.PropParams(1, epoch)
    prop_params.add_perturbation("Drag", Drag)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=atmospheric_drag,
                            R=Earth.R.to(u.km).value, C_D=C_D, A=A, m=m, H0=H0, rho0=rho0)
    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def test_ad_equals_none():
    x = [66666, 0, 0, 0, 2.451, 0]
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    t = sat_i.period.to(u.s).value
    sat_custom = sat_i.propagate(t * u.s, method=cowell, ad=None)
    sat_poli = sat_i.propagate(t * u.s, method=cowell)
    theoretical = sat_custom.r.value
    experimental = sat_poli.r.value
    assert np.array_equal(theoretical, experimental)


def a_d_j2j3(t0, state, k, J2, J3, R):
    return J2_perturbation(t0, state, k, J2, R) + J3_perturbation(t0, state, k, J3, R)


def a_d_nothing(t0, state, k):
    return atmospheric_drag(t0, state, k, Earth.R.to(u.km).value, 0, 0, 1, 1, 0)

