import numpy as np
import scipy.linalg as la
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
import astropy.units as u
from verification.util import build_observations, build_epochs, get_satellite_position_over_time
from src.interface.tle_dto import TLE
from src.enums import Frames
from src.dto import PropParams
from src.interface.local_angles import get_local_angles_via_state_propagation
from src.frames import eci_to_icrs, lla_to_ecef, ecef_to_eci
from src.observation_function import get_ra_and_dec


tle_string = """
2020-040A
1 45807U 20040A   20175.61197145 -.00000151  00000-0 -10000-3 0  9992
2 45807   0.0000 294.5058 0000000 179.9989 70.3936 2.28021460     116
"""
obs_pos = [29.218103 * u.deg, -81.031723 * u.deg, 0 * u.km]
tle = TLE.from_lines(tle_string)

x, epoch = tle.to_state()
params = PropParams(epoch)
desired_epoch = Time("2020-06-24T00:00:00.000", format="isot", scale="tdb")
dt = 1
epochs = build_epochs(desired_epoch, dt * u.h, 24)

# obj_eci, epochs = get_satellite_position_over_time(x, epochs)
# for i in range(len(epochs)):
    # obj = eci_to_icrs(positions[i], epochs[i])
    # obs = eci_to_icrs(ecef_to_eci(lla_to_ecef(obs_pos), epochs[1]), epochs[i])
    # print(get_ra_and_dec(rr))
    # print(rr)
    # print(positions[i])
    # print("norms")
    # print(la.norm(positions[i]))

# observations = build_observations(x, params, obs_pos, Frames.LLA, epochs)

# for obs in observations:
#     print(obs.obs_values)
#
n = len(epochs)
locals = get_local_angles_via_state_propagation(x, params, epochs[0], epochs[n-1], n-2, obs_pos, Frames.LLA)
for local in locals:
    local[2].format = 'isot'
    print(local)


 # Date__(UT)__HR:MN     R.A.___(ICRF)___DEC R.A._(a-appar)_DEC. Azi_(a-appr)_Elev
 # 2020-Jul-16 08:00  m   71.39000  17.78207  71.67881  17.81745  70.5231   1.7878
 # 2020-Jul-16 09:00  m   71.41659  17.78600  71.70544  17.82133  77.3749  14.3731
 # 2020-Jul-16 10:00 Nm   71.44291  17.78992  71.73179  17.82521  83.9301  27.2924
 # 2020-Jul-16 11:00 *m   71.46897  17.79381  71.75789  17.82906  90.9000  40.3707
 # 2020-Jul-16 12:00 *m   71.49483  17.79766  71.78378  17.83286  99.5451  53.4093
 # 2020-Jul-16 13:00 *m   71.52053  17.80144  71.80951  17.83660 113.2688  65.9919
 # 2020-Jul-16 14:00 *m   71.54614  17.80514  71.83514  17.84026 145.2445  76.3856
 # 2020-Jul-16 15:00 *m   71.57175  17.80876  71.86077  17.84384 210.4835  76.9496
 # 2020-Jul-16 16:00 *m   71.59743  17.81229  71.88646  17.84733 245.1981  66.9476
 # 2020-Jul-16 17:00 *m   71.62327  17.81575  71.91230  17.85074 259.6529  54.4415
 # 2020-Jul-16 18:00 *m   71.64932  17.81914  71.93836  17.85409 268.5288  41.4208
 # 2020-Jul-16 19:00 *m   71.67566  17.82248  71.96469  17.85739 275.5757  28.3407
 # 2020-Jul-16 20:00 *m   71.70232  17.82579  71.99134  17.86065 282.1376  15.4071
 # 2020-Jul-16 21:00 *m   71.72932  17.82910  72.01834  17.86391 288.9475   2.7941