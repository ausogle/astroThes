from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Moon, Sun, Earth
from poliastro.constants import Wdivc_sun, H0_earth, rho0_earth
from src.dto import ThirdBody, J2, J3, SRP, Drag
from src.constants import solar_period, lunar_period
from astropy import units as u
from astropy.time import Time


solar_system_ephemeris.set("de432s")
R = Earth.R.to(u.km).value


def build_lunar_third_body(epoch: Time, rtol=1e-2) -> ThirdBody:
    """
    This function creates a callable moon object for third_body perturbation. Over long periods of integration, this
    may longer prove to be an accurate description. Needs to be investigated.
    :param epoch:   The time about which the interpolating function is created.
    :param rtol: determines number of points generated. Drives the time of execution significantly. Example online used
    rtol=1e-2. A smaller number is not accepted. Could be increased for more accuracy, that being said, the position of
    the Moon does not need to be that accurate.
    :return: Returns callable object that describes the Moon's position
    """
    k_moon = Moon.k.to(u.km ** 3 / u.s ** 2).value
    body_moon = build_ephem_interpolant(Moon, lunar_period, (epoch.value * u.day,
                                                             epoch.value * u.day + 60 * u.day), rtol=rtol)
    return ThirdBody(k_moon, body_moon)


def build_solar_third_body(epoch: Time, rtol=1e-2) -> ThirdBody:
    """
    This function creates a callable Sun object for third_body and SRP perturbation. Over long periods of integration,
    this may longer prove to be an accurate description. Needs to be investigated.
    :param epoch:   The time about which the interpolating function is created.
    :param rtol: determines number of points generated. Drives the time of execution significantly. Example online used
    rtol=1e-2. A smaller number is not accepted. Could be increased for more accuracy, that being said, the position of
    the Sun does not need to be that accurate.
    :return: Returns callable object that describes the Sun's position
    """
    k_sun = Sun.k.to(u.km ** 3 / u.s ** 2).value
    body_sun = build_ephem_interpolant(Sun, solar_period, (epoch.value * u.day, epoch.value * u.day + 60 * u.day),
                                       rtol=rtol)
    return ThirdBody(k_sun, body_sun)


def build_j2() -> J2:
    """
    Builds J2 object used in propagation. Requires no input since all of the values are independent of orbital position
    and time.
    """
    return J2(Earth.J2.value, R)


def build_j3() -> J3:
    """
    Build J3 object used in perturbation. Requires no input since all of the values are independent of orbital position
    and time.
    """
    return J3(Earth.J3.value, R)


def build_srp(c_r, a, m, epoch, rtol=1e-2) -> SRP:
    """
    Build Solar Radiation Object used in perturbation.
    :param c_r: Comparable to coefficient of Drag but for radiation pressure. Unitless
    :param a: Cross sectional area exposed to radiation pressure. Units [m^2]
    :param m: Mass of the satellite.  Units [kg]
    :param epoch: Time required to interpolate solar position
    :param rtol: determines number of points generated. Drives the time of execution significantly. Example online used
    rtol=1e-2. A smaller number is not accepted. Could be increased for more accuracy, that being said, the position of
    the Sun does not need to be that accurate.
    """
    body_sun = build_ephem_interpolant(Sun, 1 * u.year, (epoch.value * u.day, epoch.value * u.day + 60 * u.day),
                                       rtol=rtol)
    return SRP(R, c_r, a, m, Wdivc_sun.value, body_sun)


def build_basic_drag(c_d, a, m):
    """
    Build Basic Atmospheric Drag Object used in perturbation.
    :param c_d: Coefficient of Drag. Unitless
    :param a: Cross-sectional area. Units [m^2]
    :param m: Mass. Units [kg]
    """
    return Drag(R, c_d, a, m, H0_earth, rho0_earth)
