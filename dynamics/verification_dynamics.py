import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la
from numpy import cross
from numpy.random import rand
import matplotlib.pyplot as plt
import csv

from Archives.linear_chaser_dynamics import *

# # Comparing the 2 functions depending on how we compute the derivatives of the target's position (but the vectors described are exactly the same)
# orbit = integrate.odeint(propagator_absolute,adjusted_conditions[:6],t_simulation,args=(mu,))
# orbit2 = integrate.odeint(propagator_absolute2,adjusted_conditions[:6],t_simulation,args=(mu,))
# # The two simulators give the same orbit

# # Now let's compare it to the 1st simulator of the cr3bp I've built
# orbit3 = integrate.odeint(halo_propagator,initial_conditions_bary,t_simulation,args=(mu,))
# # All these integrators give the same orbit
# # The propagator_absolute seem to be closer to the halo_propagator than propagator_absolute2

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(-orbit[:,0], -orbit[:,1], orbit[:,2], color='r', label="Target's orbit")
# ax.scatter(0, 0, 0, label='Moon')
# ax.scatter(L2x-(1-mu), 0, 0, label='L2')
# ax.axis('equal')
# ax.set_xlabel('X [nd]')
# ax.set_ylabel('Y [nd]')
# ax.set_zlabel('Z [nd]')
# plt.title("Target's orbit in the Moon (synodic) frame")
# ax.legend()
# plt.grid()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(-orbit2[:,0], -orbit2[:,1], orbit2[:,2], color='r', label="Target's orbit")
# ax.scatter(0, 0, 0, label='Moon')
# ax.scatter(L2x-(1-mu), 0, 0, label='L2')
# ax.axis('equal')
# ax.set_xlabel('X [nd]')
# ax.set_ylabel('Y [nd]')
# ax.set_zlabel('Z [nd]')
# ax.legend()
# plt.title("Target's orbit in the Moon (synodic) frame")
# plt.grid()
# plt.show()

# norm = np.zeros(len(orbit[:,0]))
# norm2 = np.zeros(len(orbit[:,0]))
# for i in range(len(orbit[:,0])):
#     error = orbit[i,:3]-orbit2[i,:3]
#     norm[i] = la.norm(error)
#     # error2 = np.zeros(3)
#     offset = np.zeros((3,1))
#     offset[0] = 1-mu
#     error2 = orbit[i,:3].reshape((3,1)) - R@(orbit3[i,:3].reshape((3,1)) - offset)
#     norm2[i] = la.norm(error2)
    
# plt.plot(t_simulation*TU/3600,norm*r12*1e3)
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$\hat{r}_{Moon} - r_{bary}$|| [m]')
# plt.title('Norm of the error between 2 propagators of the CR3BP')
# plt.show()

# plt.plot(t_simulation*TU/3600,norm2*r12*1e3)
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$r_{Moon} - r_{bary}$|| [m]')
# plt.title('Norm of the error between 2 propagators of the CR3BP')
# plt.show()