import math
from src.constants import mu
import numpy as np
from astropy.time import Time
import astropy.units as u
from verification.util import get_satellite_position_over_time
from src.frames import eci_to_ecef, ecef_to_lla, ecef_to_eci
import matplotlib.pyplot as plt

# epoch = Time("2000-01-01T00:00:00.000", format="isot", scale="utc")
# epoch.format = "jd"
day_val = 1/365.25
epoch = Time(1984, format='decimalyear', scale="utc")
period = 86164.1                     # Seconds in a sidereal day
dt = period/100
tf = period
t = np.arange(0, tf, dt)
n = t.shape[0]
a = math.pow(math.pow(period/2/math.pi, 2) * mu.value, 1/3)
speed = math.sqrt(mu.value/a)
r_0 = np.array([a, 0, 0])
x = np.array([a, 0, 0, 0, speed, 0])
# norm = np.linalg.norm(x)
# r_0eci = ecef_to_eci(r_0, epoch)
# v_0eci = np.cross(r_0eci, np.array([0, 0, speed])) / np.linalg.norm(r_0eci)
# x = np.concatenate([r_0eci, v_0eci])
# nomr2 = np.linalg.norm(x)
# x = np.array([r_0eci[0], r_0eci[1], r_0eci[2], v_0_eci[0], v_0_eci[1], v_0_eci[2]])
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

# fig = plt.figure(2)
# ax = fig.gca()
# plt.plot(t, r_lla[:, 2])
# ax.set_xlabel('time [s]')
# ax.set_ylabel('Altitude [km]')
# plt.show()

print(r_lla[:, 1])