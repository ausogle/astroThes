import math
from src.constants import mu
import numpy as np
from astropy.time import Time
import astropy.units as u
from verification.util import get_satellite_position_over_time
from src.frames import eci_to_ecef, ecef_to_lla
import matplotlib.pyplot as plt

epoch = Time(2454283.0, format="jd", scale="tdb")
period = 86164.1                     # Seconds in a sidereal day
dt = period/100
tf = period
t = np.arange(0, tf, dt)
n = t.shape[0]
a = math.pow(math.pow(period/2/math.pi, 2) * mu, 1/3)
speed = math.sqrt(mu/a)

x = np.array([a, 0, 0, 0, speed, 0])
r_eci = get_satellite_position_over_time(x, epoch, tf, dt)

r_ecef = np.zeros((n, 3))
r_lla = np.zeros((n, 3))
for i in range(0, n):
    time = epoch + t[i] * u.s
    r_ecef[i] = eci_to_ecef(r_eci[i], time)
    temp = ecef_to_lla(r_ecef[i])
    r_lla[i] = np.array([temp[0].value, temp[1].value, temp[2]. value])

fig = plt.figure(1)
ax = fig.gca()
plt.plot(r_lla[:, 1], r_lla[:, 0], 'o')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')
plt.show()

fig = plt.figure(2)
ax = fig.gca()
plt.plot(t, r_lla[:, 2])
ax.set_xlabel('time [s]')
ax.set_ylabel('Altitude [km]')
plt.show()