from src.dto import PropParams
from src.state_propagator import state_propagate
from src.interface.tle_dto import TLE

tle_string = """
STARLINK-1466
1 45732U 20038C   20192.83334491 -.01176717  00000-0 -28545-1 0  9990
2 45732  53.0016 162.9981 0001205  64.1534 251.7734 15.43303367  5600
"""

tle = TLE.from_lines(tle_string)
print("Original")
print(tle.to_string())

x, epoch = tle.to_state()
params = PropParams(epoch)
epoch_new = epoch + tle.period/2
x_new = state_propagate(x, epoch_new, params)
tle.update(x_new, epoch_new)
print("Updated")
print(tle.to_string())

