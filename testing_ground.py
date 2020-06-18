import numpy as np
from verification.util import get_period
from astropy.time import Time
import astropy.units as u
from src.state_propagator import state_propagate
from src.dto import PropParams
from src.interface.local_angles import local_angles

lla = [90 * u.s, 0 * u.s, 100 * u.km]
rr = np.array([100, 100, 0])

result = local_angles(rr, lla)
print(result)
