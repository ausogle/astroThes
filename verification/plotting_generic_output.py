from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from verification.util import generate_earth_surface


r = 6378
fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = generate_earth_surface()

u = np.linspace(0, 2*np.pi, 100)
satx = 1.5*r*np.cos(u)
saty = 1.5*r*np.sin(u)
satz = np.zeros(np.size(u))

ax.plot3D(satx, saty, satz, 'grey')
ax.plot_surface(x, y, z, color='b')

ax.set_xlim([-2*r, 2*r])
ax.set_ylim([-2*r, 2*r])
ax.set_zlim([-2*r, 2*r])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
