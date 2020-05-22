from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Moon, Sun
from src.dto import ThirdBody, ObsParams
from src.enums import Frames
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import numpy as np


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


def convert_obs_params_from_lla_to_ecef(obs_params: ObsParams):
    """
    Converts Observer location from LLA to ECEF frame before calculations to limit total computational cost.
    :param obs_params: Observational params relevant to prediction function
    """
    assert obs_params.frame == Frames.LLA
    obs_params.frame = Frames.ECEF
    obs_params.position = lla_to_ecef(obs_params.position)
    return obs_params


def lla_to_ecef(location: np.ndarray) -> np.ndarray:
    """
    Converts Lat, Lon, Alt to x,y,z position in ECEF
    :param location: List of coordinate [lat, lon, alt].
    """
    loc = EarthLocation.from_geodetic(lat=location[0], lon=location[1], height=location[2])
    r = np.array([loc.x.value, loc.y.value, loc.z.value])
    return r
