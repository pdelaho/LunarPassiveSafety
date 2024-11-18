""" This file integrates the dynamics implemented in cr3bp_barycenter.py to plot the Halo/Lyapounov/... orbit given the initial conditions.
It also uses the method of single shooting to readapt the initial conditions and get an orbit that closes after a propagation time
of one period """


import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

from cr3bp_barycenter import halo_propagator_with_STM
from single_shooting import optimization

# Data for the CR3BP with the Earth-Moon system
r12 = 389703 # km, distance between primary attractors
mu = 1.215058560962404e-2 # no unit, mass parameter of the system
TU = 382981 # s, inverse of the relative angular frequency between the two primary attractors
L1x = 0.83691513 # nd, position of the L1 point along the x direction
L2x = 1.15568217 # position of the L2 point

# Initial conditions
x0 = 1.1340389525913128E+0 # nd, for Lyapounov orbit about L1: 8.2967381582787081E-1
y0 = -1.9603275922757012E-28 # nd, for Lyapounov orbit about L1: 4.5691881617111996E-29
z0 = -3.1177930540447237E-33 # nd, for Lyapounov orbit about L1: -2.4095847440443644E-32
vx0 = 5.3802690301291542E-15 # nd, for Lyapounov orbit about L1: 2.7691850370932105E-16
vy0 = 1.1050885595242482E-1 # nd, for Lyapounov orbit about L1: 6.4159717067070993E-2
vz0 = 4.2488243310471582E-33 # nd, for Lyapounov orbit about L1: 4.2674206516771668E-32
period = 3.3898323438578979E+0 # in TU, for Lyapounov orbit about L1: 2.7041588513971861E+0

initial_STM = np.asarray(np.eye(6).flatten())
initial_state = np.concatenate(([x0, y0, z0, vx0, vy0, vz0], initial_STM))

t_simulation = np.linspace(0,period,6000)
y_orbit = integrate.odeint(halo_propagator_with_STM, initial_state, t_simulation, args=(mu,))

# Plot the orbit before single-shooting differenciation correction
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y_orbit[:,0] - (1 - mu), y_orbit[:,1], y_orbit[:,2])
ax.scatter(0, 0, 0, label='Moon')
ax.scatter(L2x - (1 - mu), 0, 0, label='L2')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.grid()

[adjusted_conditions,tf] = optimization(initial_state, period, mu)

# Plotting the orbit after adjusting initial conditions
y_orbit2 = integrate.odeint(halo_propagator_with_STM, adjusted_conditions, t_simulation, args=(mu,))

# Plot the orbit before single-shooting differenciation correction
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y_orbit2[:,0] - (1 - mu), y_orbit2[:,1], y_orbit2[:,2])
ax.scatter(0, 0, 0, label='Moon')
ax.scatter(L2x - (1 - mu), 0, 0, label='L2')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.grid()

plt.show()