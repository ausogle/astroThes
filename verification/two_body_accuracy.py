## This file serves to demonstrate the accuracy of poliastro's method for a two body scenario with no perturbations.
# Integrating force over time using the dopri8 integrator will be compared to the Lagrange/Gibbs method.
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from verification.util import *
from src.propagator import propagate
from src.dto import PropParams


r = [66666, 0, 0]
v = [0, -2.644, 0]
x = [r[0], r[1], r[2], v[0], v[1], v[2]]

a = get_a(x)
e = get_e(x)
period = get_period(x)
dt = period / 100

rr0 = r - np.zeros(len(r))
vv0 = v - np.zeros(len(r))
r0 = la.norm(rr0)
v0 = la.norm(vv0)
n = math.sqrt(mu/(a*a*a))

t = np.arange(0, period*10, dt)
F = np.zeros((len(t), 1))
G = np.zeros((len(t), 1))
r_lg = np.zeros((len(t), 3))

M = n*t

for i in range(0, len(M)):
    E = M[i]
    for j in range(0, 8):
        E = E + (M[i] - E+e*np.sin(E))/(1-e*np.cos(E))
    F[i] = 1-(a/r0)*(1-np.cos(E))
    G[i] = t[i] + math.sqrt(a*a*a/mu)*(np.sin(E)-E)
    r_lg[i] = F[i]*rr0 + G[i]*vv0

x, y, z = generate_earth_surface()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(r_lg[:, 0], r_lg[:, 1], r_lg[:, 2], 'grey')
ax.plot_surface(x, y, z, color='b')
axis_mult = 11
Re = 6378
ax.set_xlim([-axis_mult*Re, axis_mult*Re])
ax.set_ylim([-axis_mult*Re, axis_mult*Re])
ax.set_zlim([-axis_mult*Re, axis_mult*Re])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

