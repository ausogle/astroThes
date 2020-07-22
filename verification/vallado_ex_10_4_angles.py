from astropy.time import Time
import astropy.units as u
from src.enums import Frames
from src.dto import PropParams
from src.interface.local_angles import get_local_angles_via_state_propagation
from src.interface.tle_dto import TLE


# x = [5748.5350, 2679.6404, 3442.8654, 4.328274, -1.918662, -5.727629]
# epoch = Time(2449746.610150, format="jd", scale="utc")
# epoch.format = "isot"
# params = PropParams(epoch)
# epoch_i = Time("1995-01-29T02:38:37.000", format="isot", scale="utc")
# epoch_f = Time("1995-01-29T02:40:27.000", format="isot", scale="utc")
# obs_pos = [21.57 * u.deg, -158.27 * u.deg, .3002 * u.km]
#
# locals = get_local_angles_via_state_propagation(x, params, epoch_i, epoch_f, 8, obs_pos, Frames.LLA)
# for local in locals:
#     print(local)


tle_string = """
Vallado
1 45732U 20038C   20192.83334491 -.01176717  00000-0 -28545-1 0  9990
2 45732  53.0016 162.9981 0001205  64.1534 251.7734 15.43303367  5600
"""
x = [5748.5350, 2679.6404, 3442.8654, 4.328274, -1.918662, -5.727629]
epoch = Time(2449746.610150, format="jd", scale="utc")

tle = TLE.from_lines(tle_string)
tle.update(x, epoch)
tle.rev_num = 0
print(tle.to_string())