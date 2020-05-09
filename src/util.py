from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Earth, Moon, Sun
from src.dto import ThirdBody
from astropy import units as u
from astropy.time import Time


solar_system_ephemeris.set("de432s")


def build_callable_moon(epoch: Time, lunar_period=27.3, rtol=1e-2):
    """
    This function creates a callable moon object for third_body perturbation. Over long periods of integration, this
    may longer prove to be an accurate description. Needs to be investigated.
    :param epoch:   The time about which the interpolating function is created.
    :param lunar_period: Default value for the lunar period. Can be given a more accurate value if desired
    :param rtol: determines number of points generated. Drives the time of execution significantly. Example online used
    rtol=1e-2. A smaller number is not accepted. Could be increased for more accuracy, that being said, the position of
    the Moon does not need to be that accurate.
    :return: Returns callable object that describes the Moon's position
    """
    k_moon = Moon.k.to(u.km ** 3 // u.s ** 2).value
    body_moon = build_ephem_interpolant(Moon, lunar_period * u.day, (epoch.value * u.day,
                                                                     epoch.value * u.day + 60 * u.day), rtol=rtol)
    return ThirdBody(k_moon, body_moon)


def build_callable_sun(epoch: Time, rtol=1e-2):
    """
    This function creates a callable Sun object for third_body and SRP perturbation. Over long periods of integration,
    this may longer prove to be an accurate description. Needs to be investigated.
    :param epoch:   The time about which the interpolating function is created.
    :param rtol: determines number of points generated. Drives the time of execution significantly. Example online used
    rtol=1e-2. A smaller number is not accepted. Could be increased for more accuracy, that being said, the position of
    the Sun does not need to be that accurate.
    :return: Returns callable object that describes the Sun's position
    """
    k_sun = Sun.k.to(u.km ** 3 // u.s ** 2).value
    body_sun = build_ephem_interpolant(Sun, 1 * u.year, (epoch.value * u.day, epoch.value * u.day + 60 * u.day),
                                       rtol=rtol)
    return ThirdBody(k_sun, body_sun)
