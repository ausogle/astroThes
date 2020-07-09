'''
Credit to the tle-tools library on github at https://github.com/FedericoStra/tletools. I was unable to install the
package so I have copied the relevant code directly instead. There exist some small changes to the source material.
Additional functionality has been added to meet my needs.

The module :mod:`tletools.tle` defines the :class:`TLE`.
whose attributes are quantities (:class:`astropy.units.Quantity`), a type able to represent
a value with an associated unit taken from :mod:`astropy.units`.
'''

import attr
import numpy as np
import astropy.units as u
from astropy.time import Time
from poliastro.twobody import Orbit as Orbit
from poliastro.bodies import Earth as Earth
from src.interface.tle_util import parse_decimal, parse_float, m_to_nu, checksum, conv_year, DEG2RAD, RAD2DEG, rev, \
    convert_value_to_str
from src.constants import mu


@attr.s
class TLE:
    """Data class representing a single TLE.
    A two-line element set (TLE) is a data format encoding a list of orbital
    elements of an Earth-orbiting object for a given point in time, the epoch.
    All the attributes parsed from the TLE are expressed in the same units that
    are used in the TLE format.
    :ivar str name:
        Name of the satellite.
    :ivar str norad:
        NORAD catalog number (https://en.wikipedia.org/wiki/Satellite_Catalog_Number).
    :ivar str classification:
        'U', 'C', 'S' for unclassified, classified, secret.
    :ivar str int_desig:
        International designator (https://en.wikipedia.org/wiki/International_Designator),
    :ivar int epoch_year:
        Year of the epoch.
    :ivar float epoch_day:
        Day of the year plus fraction of the day.
    :ivar float dn_o2:
        First time derivative of the mean motion divided by 2.
    :ivar float ddn_o6:
        Second time derivative of the mean motion divided by 6.
    :ivar float bstar:
        BSTAR coefficient (https://en.wikipedia.org/wiki/BSTAR).
    :ivar int set_num:
        Element set number.
    :ivar float inc:
        Inclination.
    :ivar float raan:
        Right ascension of the ascending node.
    :ivar float ecc:
        Eccentricity.
    :ivar float argp:
        Argument of perigee.
    :ivar float M:
        Mean anomaly.
    :ivar float n:
        Mean motion.
    :ivar int rev_num:
        Revolution number.
    (:class:`astropy.units.Quantity`), a type able to represent a value with
    an associated unit taken from :mod:`astropy.units`.
    """

    # name of the satellite
    name = attr.ib()
    # NORAD catalog number (https://en.wikipedia.org/wiki/Satellite_Catalog_Number)
    norad = attr.ib()
    classification = attr.ib()
    int_desig = attr.ib()
    epoch_year = attr.ib(converter=conv_year)
    epoch_day = attr.ib()
    dn_o2 = attr.ib()
    ddn_o6 = attr.ib()
    bstar = attr.ib()
    set_num = attr.ib(converter=int)
    inc = attr.ib()
    raan = attr.ib()
    ecc = attr.ib()
    argp = attr.ib()
    m = attr.ib()
    n = attr.ib()
    rev_num = attr.ib(converter=int)

    @property
    def a(self):
        """Semi-major axis."""
        if self._a is None:
            self._a = (mu.value / self.n.to_value(u.rad / u.s) ** 2) ** (1 / 3) * u.m
        return self._a

    @property
    def epoch(self):
        return Time(self.epoch_year + self.epoch_day/365.25, format='decimalyear', scale='utc')

    @property
    def nu(self):
        """True anomaly."""
        if self._nu is None:
            # Make sure the mean anomaly is between -pi and pi
            m = ((self.m + 180) % 360 - 180) * DEG2RAD
            self._nu = m_to_nu(m, self.ecc) * RAD2DEG
        return self._nu

    @classmethod
    def from_lines(cls, tle_string):
        """
        Creates a tle object form a TLE string, requires name as the first line. 3 lines total

        :param tle_string: TLE string from database
        """
        tle_lines = tle_string.strip().splitlines()
        name = tle_lines[0]
        line1 = tle_lines[1]
        line2 = tle_lines[2]

        return cls(
            name=name,
            norad=line1[2:7],
            classification=line1[7],
            int_desig=line1[9:17],
            epoch_year=line1[18:20],
            epoch_day=float(line1[20:32]),
            dn_o2=str(line1[33:43]),    # altered to be a str
            ddn_o6=str(line1[44:52]),   # altered to be a str
            bstar=str(line1[53:61]),    # altered to be a str
            set_num=line1[64:68],
            inc=u.Quantity(float(line2[8:16]), u.deg),
            raan=u.Quantity(float(line2[17:25]), u.deg),
            ecc=u.Quantity(parse_decimal(line2[26:33]), u.one),
            argp=u.Quantity(float(line2[34:42]), u.deg),
            m=u.Quantity(float(line2[43:51]), u.deg),
            n=u.Quantity(float(line2[52:63]), rev / u.day),
            rev_num=line2[63:68])

    def to_orbit(self, attractor=Earth) -> Orbit:
        """Convert to an orbit around the Earth."""
        return Orbit.from_classical(
            attractor=attractor,
            a=self.a,
            ecc=self.ecc,
            inc=self.inc,
            raan=self.raan,
            argp=self.argp,
            nu=self.nu,
            epoch=self.epoch)

    def update(self, x: np.ndarray, new_epoch: Time):
        """ Updates values of the TLE according to a new estimation"""

        new_epoch.format = "decimalyear"
        r = x[0:3] * u.km
        v = x[3:6] * u.km / u.s
        obj = Orbit.from_vectors(Earth, r, v, epoch=new_epoch)
        self.epoch_year = round(new_epoch.decimalyear % 100)
        self.epoch_day = (new_epoch.decimalyear % 1) * 365.25
        self.inc = obj.inc
        self.raan = obj.raan
        self.ecc = obj.ecc
        self.argp = obj.argp
        self.m = obj.M
        self.n = (mu.value / (self.a.value ** 3)) ** (1/2)
        self.set_num += 1

        return self

    def to_string(self):
        """Convert TLE into a string"""

        self.epoch.format = "decimalyear"
        epoch_year = str(int(self.epoch.decimalyear))[2:4]
        epoch_day = str(convert_value_to_str((self.epoch.value % 1) * 365.25, 3, 8))

        set_num = convert_value_to_str(self.set_num, 4, 0)[0:4]
        inc = convert_value_to_str(self.inc.to(u.deg).value, 3, 4)
        raan = convert_value_to_str(self.raan.to(u.deg).value, 3, 4)
        ecc = convert_value_to_str(self.ecc.value, 1, 7)[2:]
        argp = convert_value_to_str(self.argp.to(u.deg).value, 3, 4)
        m = convert_value_to_str(self.m.to(u.deg).value, 3, 4)
        n = convert_value_to_str(self.n.value, 2, 8)
        rev_num = convert_value_to_str(self.rev_num, 5, 0)[0:5]

        line0 = self.name
        line1 = "\n1 " + self.norad + self.classification + " " + self.int_desig + " " + epoch_year + epoch_day + " " \
                + self.dn_o2 + " " + self.ddn_o6 + " " + self.bstar + " 0 " + set_num
        line2 = "\n2 " + self.norad + " " + inc + " " + raan + " " + ecc + " " + argp + " " + m + " " + n + rev_num
        return line0 + line1 + checksum(line1[1:]) + line2 + checksum(line2[1:])
