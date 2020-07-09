import numpy as np
from tletools import TLE
from src.dto import PropParams
from typing import Tuple
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import astropy.units as u
from src.constants import mu
import string
from astropy.time import Time


def tle_to_state(tle_string: str) -> Tuple[np.ndarray, PropParams]:
    """
    Converts a tle into a state vector and PropParams object that holds the epoch

    :param tle_string: TLE string
    """
    tle_lines = tle_string.strip().splitlines()
    tle = TLE.from_lines(*tle_lines)
    sat = tle.to_orbit()

    epoch_yr = float(tle_lines[1][18:20])
    epoch_day = float(tle_lines[1][20:32])
    if epoch_yr < 57:
        year = 2000 + epoch_yr
    else:
        year = 1900 + epoch_yr
    epoch = Time(year + epoch_day / 365.25, format="decimalyear", scale="utc")

    x = np.concatenate([sat.r.value, sat.v.value])
    prop_params = PropParams(epoch)
    return x, prop_params


def state_to_tle(tle_string: str, x: np.ndarray, params: PropParams) -> str:
    """
    Converts a state vector and epoch to a TLE. Note: Is inaccurate with respect to epoch and RAAN. From what I can tell
    this is due to issues inside of Poliastro. I have checked their math and I would go about it the same way. The ideal
    solution would be to build a custom Time format and handle it that way. As far as RAAN, no clue.

    :param tle_string: The TLE string
    :param x: presumably more correct state vector
    :param params: PropParams object that holds epoch of new state vector
    """
    r = x[0:3] * u.km
    v = x[3:6] * u.km / u.s
    obj = Orbit.from_vectors(Earth, r, v, epoch=params.epoch)

    tle_lines = tle_string.strip().splitlines()
    tle = TLE.from_lines(*tle_lines)

    name = tle_lines[0]
    norad = tle_lines[1][2:7]
    classification = tle_lines[1][7]
    int_desig = tle_lines[1][9:17]

    epoch_value = obj.epoch.decimalyear
    epoch_year = str(epoch_value)[2:4]
    epoch_day_value = (epoch_value % 1) * 365.25
    epoch_day = convert_value_to_str(epoch_day_value, 3, 8)

    dn_02 = tle_lines[1][33:43]
    ddn_06 = tle_lines[1][44:52]
    bstar = tle_lines[1][53:61]
    set_num = tle_lines[1][64:68]
    rev_num_i = tle_lines[2][63:68]
    epoch_i = tle.epoch

    inc = convert_value_to_str(obj.inc.to(u.deg).value, 3, 4)
    raan = convert_value_to_str(obj.raan.to(u.deg).value, 3, 4)
    ecc = convert_value_to_str(obj.ecc.value, 1, 7)[2:]
    argp = convert_value_to_str(obj.argp.to(u.deg).value, 3, 4)

    a = obj.a.value
    n_val = np.sqrt(mu.value / (a * a * a)) / (2 * np.pi) * 86400
    n = str(n_val)[0:11]

    epoch_f = obj.epoch
    dt = (epoch_f - epoch_i).to(u.s)
    period = 2 * np.pi * np.sqrt((a * a * a) / mu.value)
    new_revs = int(rev_num_i) + np.floor((dt / period).value)
    total_revs = convert_value_to_str(new_revs, 5, 0)[0:5]

    m_value = nu_to_M(obj.nu.to(u.deg), obj.ecc)
    m = convert_value_to_str(m_value, 3, 4)

    line0 = name
    line1 = "\n1 " + norad + classification + " " + int_desig + " " + epoch_year + epoch_day + " " + dn_02 \
            + " " + ddn_06 + " " + bstar + " 0 " + set_num
    line2 = "\n2 " + norad + " " + inc + " " + raan + " " + ecc + " " + argp + " " + m + " " + n + total_revs
    return line0 + line1 + checksum(line1[1:]) + line2 + checksum(line2[1:])


def convert_value_to_str(value, length_before, length_after) -> str:
    """
    Converts a value into a string with appropriate formatting. Allows user to decide how many characters are before and
    after the decimal point.

    :param value: the value to be formatted
    :param length_before: Number of characters before the decimal
    :param length_after: Number of characters after the decimal
    """
    after = value % 1
    before = value - after
    before_string = str(int(before))
    after_string = str(after)[2:]
    while len(before_string) < length_before:
        before_string = " " + before_string
    while len(after_string) < length_after:
        after_string += "0"
    return before_string[0:length_before] + "." + after_string[0:length_after]


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
