""" This file compares the trajectory of the chaser spacecraft when integrating the linear dynamics and when using the 
state transition matrix (STM) """


import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la
from numpy import cross
from numpy.random import rand
import matplotlib.pyplot as plt
import csv
import scipy as sc


from linear_dynamics_LVLH import propagator_relative, integrate_matrix

# Data for the CR3BP with the Earth-Moon system
r12 = 384400 # km, distance between primary attractors, from JPL website 389703
mu = 1.215e-2 # no unit, mass parameter of the system, from JPL website 1.215058560962404e-2
TU = 1/(2.661699e-6) # s, inverse of the relative angular frequency between the two primary attractors, from JPL website 382981
L1x = 0.83691513 # nd, position of the L1 point along the x direction
L2x = 1.15568217 # nd, position of the L2 point along the x direction

# Rotation matrix between barycenter and Moon frame (there is also a translation component, taken care of later)
R = np.array([
    [-1, 0, 0], 
    [0, -1, 0], 
    [0, 0, 1]
    ])

# Initial position w.r.t. the barycenter of the Earth-Moon system in the bary frame, current conditions for a L2 Southern NHRO
x0_bary = 1.0269694420519750E+0 #   nd 1.1340389525913128E+0,   southern L1 NHRO 9.2280005557282274E-1
y0_bary = -1.0620425026230252E-26 # nd -1.9603275922757012E-28, southern L1 NHRO1.6386233489716853E-28
z0_bary = -1.8530606468865049E-1 #  nd -3.1177930540447237E-33, southern L1 NHRO -2.1575768509057866E-1

# Initial position w.r.t. the barycenter of the Earth-Moon system in the Moon frame
init_pos_M = R @ np.array([x0_bary - (1 - mu), y0_bary, z0_bary]).reshape((3,1))
x0_M, y0_M, z0_M = init_pos_M[:3,0]

# Initial velocity w.r.t. the barycenter of the Earth-Moon system in the bary frame
vx0_bary = 1.8339007690300910E-14 #  nd 5.3802690301291542E-15, southern L1 NHRO 4.4327633188679963E-13
vy0_bary = 	-1.1378551488655682E-1 # nd 1.1050885595242482E-1,  southern L1 NHRO 1.2826547451754347E-1
vz0_bary = 1.3151545077882733E-13 #  nd 4.2488243310471582E-33, southern L1 NHRO 2.4299327620081873E-12

# Initial velocity w.r.t. the barycenter of the Earth-Moon system in the Moon frame
init_vel_M = R @ np.array([vx0_bary, vy0_bary, vz0_bary]).reshape((3,1))
vx0_M, vy0_M, vz0_M = init_vel_M[:3,0]

# Period of the corresponding orbit
period = 1.5763752384473892E+0 # nd 3.3898323438578979E+0, southern L1 NHRO 1.8036720655626510E+0

# Initial conditions in the Moon and barycenter frames for the target spacecraft
init_cond_M = [x0_M, y0_M, z0_M, vx0_M, vy0_M, vz0_M]
init_cond_bary = [x0_bary, y0_bary, z0_bary, vx0_bary, vy0_bary, vz0_bary]

# Seeting initial conditions for the chaser in the LVLH frame
dist_to_target_km = 10 # in km
dist_to_target = dist_to_target_km / r12

rho_x0_LVLH = rand() * dist_to_target
rho_y0_LVLH = rand() * np.sqrt(dist_to_target**2 - rho_x0_LVLH**2)
rho_z0_LVLH = np.sqrt(dist_to_target**2 - rho_x0_LVLH**2 - rho_y0_LVLH**2)

vel_rel_target_km = 0 # in km/s
vel_rel_target = vel_rel_target_km / r12 * TU

rho_vx0_LVLH = rand() * vel_rel_target
rho_vy0_LVLH = rand() * np.sqrt(vel_rel_target**2 - rho_vx0_LVLH**2)
rho_vz0_LVLH = np.sqrt(vel_rel_target**2 - rho_vx0_LVLH**2 - rho_vy0_LVLH**2)

init_cond_chaser_LVLH = [rho_x0_LVLH, rho_y0_LVLH, rho_z0_LVLH, rho_vx0_LVLH, rho_vy0_LVLH, rho_vz0_LVLH]

# Initial conditions to integrate the relative dynamics
initial_conditions_chaser_LVLH = np.concatenate((init_cond_M, init_cond_chaser_LVLH))

# Simulating the relative dynamics for 12 hours
length = 12 * 3600 / TU
t_simulation = np.linspace(0, length, 1000)
y_orbit = integrate.odeint(propagator_relative, initial_conditions_chaser_LVLH, t_simulation, args=(mu,))

# Plotting of the target's orbit in the Moon frame
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(- y_orbit[:, 0], - y_orbit[:, 1], y_orbit[:, 2], color='r', label="Target's orbit")
ax.scatter(0, 0, 0, label='Moon')
ax.scatter(L2x - (1 - mu), 0, 0, label='L2')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Target's orbit in the Moon frame")
plt.grid()

# Plotting of the chaser's position in the LVLH frame
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y_orbit[:, 6] * r12, y_orbit[:, 7] * r12, y_orbit[:, 8] * r12, color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Target spacecraft')
ax.axis('equal')
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame")
plt.grid()

# Computing the chaser's trajectory when using the STM to compute the state derivative
matrix_dyn = integrate.odeint(integrate_matrix, initial_conditions_chaser_LVLH, t_simulation, args=(mu,))

# Plotting of the chaser's position in the LVLH frame
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(matrix_dyn[:, 6] * r12, matrix_dyn[:, 7] * r12, matrix_dyn[:, 8] * r12, color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Target spacecraft')
ax.axis('equal')
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame")
plt.grid()

# Computing the difference between the position given by integrating linear dynamics and dynamics obtained with STM
error_mat_dist = np.zeros(y_orbit.shape[0])
error_mat_vel = np.zeros(y_orbit.shape[0])
for i in range(y_orbit.shape[0]):
    error_mat_dist[i] = la.norm(matrix_dyn[i, 6:9] - y_orbit[i, 6:9])
    error_mat_vel[i] = la.norm(matrix_dyn[i, 10:12] - y_orbit[i, 10:12])
    
plt.figure()
plt.plot(t_simulation * TU / 3600, error_mat_dist * r12 * 1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\rho_{STM,int} - \rho_{int}$|| [m]")
plt.title('Distance error between integration of linear dynamics and integration of linear with matrix form')

plt.figure()
plt.plot(t_simulation * TU / 3600, error_mat_vel * r12 * 1e3 / TU)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\dot{\rho}_{STM,int} - \dot{\rho}_{int}$|| [m/s]")
plt.title('Velocity error between integration of linear dynamics and integration of linear with matrix form')

plt.show()