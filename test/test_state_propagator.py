import numpy as np
from src.state_propagator import a_d, state_propagate
from src.util import build_j2, build_j3, build_basic_drag, build_lunar_third_body, build_solar_third_body, build_srp
from poliastro.ephem import build_ephem_interpolant
from poliastro.bodies import Moon, Sun
from astropy.coordinates import solar_system_ephemeris
from src.enums import Perturbations
from src.constants import lunar_period, solar_period
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation, J3_perturbation, atmospheric_drag_exponential, third_body, \
    radiation_pressure
from poliastro.constants import H0_earth, rho0_earth, Wdivc_sun
from astropy import units as u
from astropy.time import Time
from src.dto import PropParams


solar_system_ephemeris.set("de432s")
R = Earth.R.to(u.km).value


def test_propagate_with_j2j3():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100 * u.day
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_obs = epoch + dt

    prop_params = PropParams(epoch)
    prop_params.add_perturbation(Perturbations.J2, build_j2())
    prop_params.add_perturbation(Perturbations.J3, build_j3())

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt, method=cowell, ad=a_d_j2j3, J2=Earth.J2.value, R=R, J3=Earth.J3.value)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(x, epoch_obs, prop_params)
    assert np.array_equal(x_custom, x_poli)


def test_propagate_with_drag():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100 * u.s
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_obs = epoch + dt
    C_D = 1
    A = 10
    m = 1000
    Drag = build_basic_drag(C_D, A, m)
    prop_params = PropParams(epoch)
    prop_params.add_perturbation(Perturbations.Drag, Drag)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt, method=cowell, ad=atmospheric_drag_exponential, R=R, C_D=C_D, A=A, m=m, H0=H0_earth,
                            rho0=rho0_earth)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(x, epoch_obs, prop_params)
    assert np.array_equal(x_custom, x_poli)


def test_propagate_with_no_perturbations():
    x = [66666, 0, 0, 0, 2.451, 0]
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    dt = 100 * u.day
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_obs = epoch + dt
    prop_params = PropParams(epoch)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt, method=cowell)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(x, epoch_obs, prop_params)
    assert np.array_equal(x_custom, x_poli)


def test_propagate_with_lunar_third_body():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 8600
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_f = epoch + (dt * u.s)
    prop_params = PropParams(epoch)
    prop_params.add_perturbation(Perturbations.Moon, build_lunar_third_body(epoch))

    k_moon = Moon.k.to(u.km ** 3 / u.s ** 2).value
    body_moon = build_ephem_interpolant(Moon, lunar_period, (epoch.value * u.day,
                                                             epoch.value * u.day + 60 * u.day), rtol=1e-2)
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=third_body, k_third=k_moon, third_body=body_moon)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(np.array(x), epoch_f, prop_params)
    assert np.array_equal(x_custom, x_poli)


def test_propagate_with_solar_third_body():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 1 * u.day
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_obs = epoch + dt
    prop_params = PropParams(epoch)
    prop_params.add_perturbation(Perturbations.Sun, build_solar_third_body(epoch))

    k_sun = Sun.k.to(u.km ** 3 / u.s ** 2).value
    body_sun = build_ephem_interpolant(Sun, solar_period, (epoch.value * u.day,
                                                           epoch.value * u.day + 60 * u.day), rtol=1e-2)
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt, method=cowell, ad=third_body, k_third=k_sun, third_body=body_sun)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(x, epoch_obs, prop_params)
    assert np.array_equal(x_custom, x_poli)


def test_propagate_with_srp():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 1 * u.day
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    epoch_obs = epoch + dt
    C_R = 1
    A = 10
    m = 1000
    srp = build_srp(C_R, A, m, epoch)
    prop_params = PropParams(epoch)
    prop_params.add_perturbation(Perturbations.SRP, srp)

    body_sun = build_ephem_interpolant(Sun, solar_period, (epoch.value * u.day,
                                                           epoch.value * u.day + 60 * u.day), rtol=1e-2)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt, method=cowell, ad=radiation_pressure,
                            R=R, C_R=C_R, A=A, m=m, Wdivc_s=Wdivc_sun.value, star=body_sun)
    x_poli = np.concatenate([sat_f.r.value, sat_f.v.value])

    x_custom = state_propagate(x, epoch_obs, prop_params)
    assert np.array_equal(x_custom, x_poli)


def a_d_j2j3(t0, state, k, J2, J3, R):
    return J2_perturbation(t0, state, k, J2, R) + J3_perturbation(t0, state, k, J3, R)

