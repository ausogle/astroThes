from astropy.time import Time
import astropy.units as u
from src.enums import Frames
from src.dto import PropParams
from src.interface.local_angles import get_local_angles_via_state_propagation


x = [5748.5350, 2679.6404, 3442.8654, 4.328274, -1.918662, -5.727629]
epoch = Time(2449746.610150, format="jd", scale="utc")
epoch.format = "isot"
params = PropParams(epoch)
epoch_i = Time("1995-01-29T02:38:37.000", format="isot", scale="utc")
epoch_f = Time("1995-01-29T02:40:27.000", format="isot", scale="utc")
obs_pos = [21.57 * u.deg, -158.27 * u.deg, .3002 * u.km]

locals = get_local_angles_via_state_propagation(x, params, epoch_i, epoch_f, 8, obs_pos, Frames.LLA)
for local in locals:
    print(local)
