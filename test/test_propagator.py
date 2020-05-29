import numpy as np
from src.propagator import a_d, propagate
from src.util import build_j2, build_j3, build_basic_drag, build_lunar_third_body, build_solar_third_body, build_srp
from poliastro.ephem import build_ephem_interpolant
from poliastro.bodies import Moon, Sun
from astropy.coordinates import solar_system_ephemeris
from src.enums import Perturbations
from src.constants import lunar_period, solar_period
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation, J3_perturbation, atmospheric_drag, third_body, radiation_pressure
from poliastro.constants import H0_earth, rho0_earth, Wdivc_sun
from astropy import units as u
from astropy.time import Time
from src import dto


solar_system_ephemeris.set("de432s")
R = Earth.R.to(u.km).value


def test_propagate_with_j2j3():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")

    prop_params = dto.PropParams(1, epoch)
    prop_params.add_perturbation(Perturbations.J2, build_j2())
    prop_params.add_perturbation(Perturbations.J3, build_j3())

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=a_d_j2j3, J2=Earth.J2.value, R=R, J3=Earth.J3.value)
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
    Drag = build_basic_drag(C_D, A, m)
    prop_params = dto.PropParams(dt, epoch)
    prop_params.add_perturbation(Perturbations.Drag, Drag)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=atmospheric_drag, R=R, C_D=C_D, A=A, m=m, H0=H0_earth, rho0=rho0_earth)
    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def test_propagate_with_no_perturbations():
    x = [66666, 0, 0, 0, 2.451, 0]
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    t = sat_i.period.to(u.s).value
    prop_params = dto.PropParams(t, epoch)
    result = propagate(x, prop_params)
    sat_poli = sat_i.propagate(t * u.s, method=cowell)
    theoretical = result[0:3]
    experimental = sat_poli.r.value
    assert np.array_equal(theoretical, experimental)


def test_propagate_with_lunar_third_body():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    prop_params = dto.PropParams(dt, epoch)
    prop_params.add_perturbation(Perturbations.Moon, build_lunar_third_body(epoch))

    k_moon = Moon.k.to(u.km ** 3 / u.s ** 2).value
    body_moon = build_ephem_interpolant(Moon, lunar_period, (epoch.value * u.day,
                                                             epoch.value * u.day + 60 * u.day), rtol=1e-2)
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=third_body, k_third=k_moon, third_body=body_moon)
    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def test_propagate_with_solar_third_body():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    prop_params = dto.PropParams(dt, epoch)
    prop_params.add_perturbation(Perturbations.Sun, build_solar_third_body(epoch))

    k_sun = Sun.k.to(u.km ** 3 / u.s ** 2).value
    body_sun = build_ephem_interpolant(Sun, solar_period, (epoch.value * u.day,
                                                           epoch.value * u.day + 60 * u.day), rtol=1e-2)
    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=third_body, k_third=k_sun, third_body=body_sun)
    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def test_propagate_with_srp():
    x = [66666, 0, 0, 0, 2.451, 0]
    dt = 100
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    epoch = Time(2454283.0, format="jd", scale="tdb")
    C_R = 1
    A = 10
    m = 1000
    srp = build_srp(C_R, A, m, epoch)
    prop_params = dto.PropParams(dt, epoch)
    prop_params.add_perturbation(Perturbations.SRP, srp)

    body_sun = build_ephem_interpolant(Sun, solar_period, (epoch.value * u.day,
                                                           epoch.value * u.day + 60 * u.day), rtol=1e-2)

    sat_i = Orbit.from_vectors(Earth, r, v, epoch=epoch)
    sat_f = sat_i.propagate(dt * u.s, method=cowell, ad=radiation_pressure,
                            R=R, C_R=C_R, A=A, m=m, Wdivc_s=Wdivc_sun.value, star=body_sun)

    output_f = np.concatenate([sat_f.r.value, sat_f.v.value])
    sat_custom = sat_i.propagate(dt * u.s, method=cowell, ad=a_d, perturbations=prop_params.perturbations)
    output_custom = np.concatenate([sat_custom.r.value, sat_custom.v.value])
    assert np.array_equal(output_custom, output_f)


def a_d_j2j3(t0, state, k, J2, J3, R):
    return J2_perturbation(t0, state, k, J2, R) + J3_perturbation(t0, state, k, J3, R)


def a_d_nothing(t0, state, k):
    return atmospheric_drag(t0, state, k, Earth.R.to(u.km).value, 0, 0, 1, 1, 0)
