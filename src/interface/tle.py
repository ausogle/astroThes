import numpy as np
from tletools import TLE
from src.dto import PropParams
from typing import Tuple


def tle_to_state(tle_string: str) -> Tuple[np.ndarray, PropParams]:
    tle_lines = tle_string.strip().splitlines()
    tle = TLE.from_lines(*tle_lines)
    sat = tle.to_orbit()
    x = np.concatenate([sat.r.value, sat.v.value])
    prop_params = PropParams(sat.epoch)
    return x, prop_params


# NEED STATE (AND RELEVANT PARAMETERS SUCH AS EPOCH AND SA) TO TLE FUNCTION
