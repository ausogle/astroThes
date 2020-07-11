import string
import numpy as np
import astropy.units as u
from poliastro.core.angles import M_to_E, E_to_nu, nu_to_E, E_to_M

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi

rev = u.def_unit(
    ['rev', 'revolution'],
    2.0 * np.pi * u.rad,
    prefixes=False,
    doc="revolution: angular measurement, a full turn or rotation")
u.add_enabled_units(rev)


def conv_year(s):
    """Interpret a two-digit year string."""
    if isinstance(s, int):
        return s
    y = int(s)
    return y + (1900 if y >= 57 else 2000)


def parse_decimal(s):
    """Parse a floating point with implicit leading dot.
    """
    return float('.' + s)


def parse_float(s):
    """Parse a floating point with implicit dot and exponential notation.
    """
    return float(s[0] + '.' + s[1:6] + 'e' + s[6:8])


def m_to_nu(m, ecc):
    """True anomaly from mean anomaly.
    :param float m: Mean anomaly in radians.
    :param float ecc: Eccentricity.
    :returns: `nu`, the true anomaly, between -π and π radians.

    **Warning**
    The mean anomaly must be between -π and π radians.
    The eccentricity must be less than 1.
    """
    return E_to_nu(M_to_E(m, ecc), ecc)


def nu_to_m(nu, ecc):
    """Mean anomaly from true anomaly.
    :param float nu: True anomaly in radians.
    :param float ecc: Eccentricity.
    :returns: `nu`, the true anomaly, between -π and π radians.

    **Warning**
    The mean anomaly must be between -π and π radians.
    The eccentricity must be less than 1.
    """
    return E_to_M(nu_to_E(nu.to(u.rad), ecc), ecc) * u.rad


def checksum(line: str) -> str:
    """
    Check sum function for TLEs. Adds all non-letters as their value and - signs as 1.

    :param line: A line of a TLE
    """
    L = line.strip()
    cksum = 0
    for i in range(68):
        c = L[i]
        if c == " " or c == "." or c == "+" or c in string.ascii_letters:
            continue
        elif c == "-":
            cksum = cksum + 1
        else:
            cksum = cksum + int(c)
    cksum %= 10
    return str(cksum)


def convert_value_to_str(value, length_before, length_after) -> str:
    """
    Converts a value into a string with appropriate formatting. Allows user to decide how many characters are before and
    after the decimal point.

    :param value: the value to be formatted
    :param length_before: Number of characters before the decimal
    :param length_after: Number of characters after the decimal
    """
    thing = value % 1
    after = round(value % 1, length_after)
    before = round(value - after)
    before_string = str(int(before))
    after_string = str(after)[2:]
    while len(before_string) < length_before:
        before_string = " " + before_string
    while len(after_string) < length_after:
        after_string += "0"
    output = before_string[0:length_before] + "." + after_string[0:length_after]
    return output


def ensure_positive_angle(angle):
    angle = angle.to(u.deg)
    while angle.value < 0:
        angle += (360 * u.deg)
    while angle.value > 360:
        angle -= (360 * u.deg)
    return angle
