from astropy.coordinates import GCRS, ITRS, ICRS, CIRS, CartesianRepresentation, EarthLocation
from astropy import units as u
from astropy.time import Time
import numpy as np
from typing import List


def lla_to_ecef(lla: List) -> np.ndarray:
    """
    Converts Lat, Lon, Alt to x,y,z position in ECEF
    :param lla: List of coordinate [lat, lon, alt]. Units [deg, deg, km]
    """
    loc = EarthLocation.from_geodetic(lat=lla[0], lon=lla[1], height=lla[2])
    r = np.array([loc.x.value, loc.y.value, loc.z.value])
    return r


def ecef_to_lla(r: np.ndarray) -> List:
    """
    Converts coordinate in ECEF frame to Lat and Lon
    :param r: Position in ECEF frame. Numpy array Units [km]
    """
    loc = EarthLocation.from_geocentric(r[0] * u.km, r[1] * u.km, r[2] * u.km)
    lat = loc.geodetic.lat
    lon = loc.geodetic.lon
    alt = loc.geodetic.height
    lla = [lat, lon, alt]
    return lla


def eci_to_ecef(r: np.ndarray, time: Time) -> np.ndarray:
    """
    Converts coordinates in Earth Centered Inertial frame to Earth Centered Earth Fixed.
    :param r: position of satellite in ECI frame. Units [km]
    :param time: Time of observation
    """
    gcrs = GCRS(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km), obstime=time)
    itrs = gcrs.transform_to(ITRS(obstime=time))
    x_ecef = itrs.x.value
    y_ecef = itrs.y.value
    z_ecef = itrs.z.value
    return np.array([x_ecef, y_ecef, z_ecef])


def ecef_to_eci(r: np.ndarray, time: Time) -> np.ndarray:
    """
    Converts coordinates in Earth Centered Earth Initial frame to Earth Centered Initial.
    :param r: position of satellite in ECEF frame. Units [km]
    :param time: Time of observation
    """
    itrs = ITRS(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km), obstime=time)
    gcrs = itrs.transform_to(GCRS(obstime=time))
    x_eci = gcrs.cartesian.x.value
    y_eci = gcrs.cartesian.y.value
    z_eci = gcrs.cartesian.z.value
    return np.array([x_eci, y_eci, z_eci])


def eci_to_icrs(r: np.ndarray, time: Time) -> np.ndarray:
    gcrs = GCRS(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km), obstime=time)
    icrs = gcrs.transform_to(ICRS)
    x_icrs = icrs.cartesian.x.value
    y_icrs = icrs.cartesian.y.value
    z_icrs = icrs.cartesian.z.value
    return np.array([x_icrs, y_icrs, z_icrs])


def icrs_to_eci(r: np.ndarray, time: Time) -> np.ndarray:
    icrs = ICRS(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km))
    gcrs = icrs.transform_to(GCRS)
    x_gcrs = gcrs.cartesian.x.value
    y_gcrs = gcrs.cartesian.y.value
    z_gcrs = gcrs.cartesian.z.value
    return np.array([x_gcrs, y_gcrs, z_gcrs])
