from astropy.time import Time
from src.dto import PropParams
from src.interface.tle_dto import TLE
import numpy as np


x = np.array([5748.5350, 2679.6404, 3442.8654, 4.328274, -1.918662, -5.727629])
epoch = Time("2000-01-01T10:10:10.101", format="isot", scale="utc")

tle_string = """
Fake Tle
1 00001U 00001A   20175.61197145 -.00000151  00000-0 -10000-3 0  9998
2 00001  28.4194 294.5058 7294772 179.9989 70.3936 2.28021460    11
"""

tle = TLE.from_lines(tle_string)
print("Original")
print(tle.to_string())

print("Updated")
tle.update(x, epoch)
print(tle.to_string())