from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy import units as u
from astropy.time import Time
import numpy as np
from typing import List


def lla_to_ecef(location: List) -> np.ndarray:
    """
    Converts Lat, Lon, Alt to x,y,z position in ECEF
    :param location: List of coordinate [lat, lon, alt]. Units [deg, deg, km]
    """
    loc = EarthLocation.from_geodetic(lat=location[0], lon=location[1], height=location[2])
    r = np.array([loc.x.value, loc.y.value, loc.z.value])
    return r


def eci_to_ecef(r: np.ndarray, time: Time):
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
