# Run this file when you want to change the initial conditions used in the other file
# See the target_orbits.txt to see previous orbital initial conditions used for the target's orbit

import numpy as np
import csv
from numpy.random import rand
from numpy import cross
import scipy
import scipy.integrate
import random

from linear_dynamics_LVLH import *

# General data for the CR3BP
r12 = 384400 # km, distance between primary attractors, from JPL website 389703
mu = 1.215e-2 # no unit, mass parameter of the system, from JPL website 1.215058560962404e-2
TU = 1/(2.661699e-6) # s, inverse of the relative angular frequency between the two primary attractors, from JPL website 382981
L1x = 0.83691513 # nd, position of the L1 point along the x direction
L2x = 1.15568217 # nd, position of the L2 point along the x direction

R = np.matrix([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]) # rotation matrix to go from the bary to the Moon frame and conversely (+ translation along x)

# Initial position w.r.t. the barycenter of the Earth-Moon system in the bary frame
x0_bary = 9.2280005557282274E-1 # nd 
y0_bary = 1.6386233489716853E-28 # nd 
z0_bary = -2.1575768509057866E-1 # nd 

# nitial position w.r.t. the barycenter of the Earth-Moon system in the Moon frame
initial_position_M = R@np.matrix([x0_bary-(1-mu), y0_bary, z0_bary]).reshape((3,1))
x0_M = initial_position_M[0,0]
y0_M = initial_position_M[1,0]
z0_M = initial_position_M[2,0]

# Initial velocity w.r.t. the barycenter of the Earth-Moon system in the bary frame
vx0_bary = 4.4327633188679963E-13 # nd 
vy0_bary = 	1.2826547451754347E-1 # nd 
vz0_bary = 2.4299327620081873E-12 # nd 

# Initial velocity w.r.t. the barycenter of the Earth-Moon system in the Moon frame
initial_velocity_M = R@np.matrix([vx0_bary, vy0_bary, vz0_bary]).reshape((3,1))
vx0_M = initial_velocity_M[0,0]
vy0_M = initial_velocity_M[1,0]
vz0_M = initial_velocity_M[2,0]
period = 1.8036720655626510E+0 # nd 

# Initial conditions in the Moon frame for the target spacecraft
initial_conditions_M = [x0_M, y0_M, z0_M, vx0_M, vy0_M, vz0_M]

# Add a part to change the mean anomaly at which the target spacecraft starts
# Assuming that the first initial conditions to get the target's orbit are at the apoapsis, ie M = 180°
M = np.radians(340) # TO CHANGE TO TRY DIFFERENT INITIAL CONDITIONS
# Using M = 2*pi*t/T where T is the orbit period
# Not sure if the following line is the correct
t = M*period/(2*np.pi) - period/2 # time at which we need to stop the simulation to get the "new" initial conditions
t_IC = np.linspace(0,t,1000)

position = scipy.integrate.odeint(propagator_absolute,initial_conditions_M,t_IC,args=(mu,))
# print(position[-1,:],initial_conditions_M)
initial_conditions_M = position[-1,:]
x0_M = initial_conditions_M[0]
y0_M = initial_conditions_M[1]
z0_M = initial_conditions_M[2]
vx0_M = initial_conditions_M[3]
vy0_M = initial_conditions_M[4]
vz0_M = initial_conditions_M[5]

# Initial conditions in the bary frame for the target spacecraft
# initial_conditions_bary = [x0_bary, y0_bary, z0_bary, vx0_bary, vy0_bary, vz0_bary]
pos = (R.T @ np.asarray([x0_M,y0_M,z0_M]).reshape((3,1)) + np.asarray([1-mu,0,0]).reshape((3,1))).reshape(3)
# print(pos,pos[0,0])
vel = (R.T @ np.asarray([vx0_M,vy0_M,vz0_M]).reshape((3,1))).reshape(3)
# print(vx,vy,vz)
initial_conditions_bary = [pos[0,0],pos[0,1],pos[0,2],vel[0,0],vel[0,1],vel[0,2]]
# print(initial_conditions_bary)

# Setting the initial conditions for the chaser. We'll use the same orbit as the precedent one for the target.
# Initial conditions for the chaser are defined in the LVLH frame

# Maybe do that in another file and save it in a csv file so that the initial conditions don't change everytime we run the code here
# So that we can actually compare results with the same initial conditions more easily

distance_to_target_km = 10 # in km, CHANGE to see to which extent the linear approximation works (try up to 100km in the article)
distance_to_target = distance_to_target_km/r12 # adimensionalized initial condition

rho_x0_LVLH = rand()*distance_to_target*random.choice([1,-1]) # NOT ENTIRELY RANDOM SINCE IT CAN'T BE NEGATIVE, CHANGE THAT!!! -> is it relevant?
rho_y0_LVLH = rand()*np.sqrt(distance_to_target**2 - rho_x0_LVLH**2)*random.choice([1,-1])
rho_z0_LVLH = np.sqrt(distance_to_target**2 - rho_x0_LVLH**2 - rho_y0_LVLH**2)*random.choice([1,-1])

velocity_rel_target_km = 0 # in km/s, CHANGE to see to which extent the linear approximation works
velocity_rel_target = velocity_rel_target_km/r12*TU # adimensionalized initial condition

rho_vx0_LVLH = rand()*velocity_rel_target
rho_vy0_LVLH = rand()*np.sqrt(velocity_rel_target**2 - rho_vx0_LVLH**2)
rho_vz0_LVLH = np.sqrt(velocity_rel_target**2 - rho_vx0_LVLH**2 - rho_vy0_LVLH**2)
# print(rho_vx0_LVLH,rho_vy0_LVLH,rho_vz0_LVLH)

# Initial conditions to integrate the relative dynamics
# The initial conditions are in the Moon frame for the target part and in the LVLH frame for the chaser part
initial_conditions_chaser_LVLH = [x0_M, y0_M, z0_M, vx0_M, vy0_M, vz0_M, rho_x0_LVLH, rho_y0_LVLH, rho_z0_LVLH, rho_vx0_LVLH, rho_vy0_LVLH, 
                                  rho_vz0_LVLH]

# Computing the initial LVLH frame to get the rotation matrix and get the initial conditions in the Moon synodic frame
r_M_init = np.asarray([[x0_M],[y0_M],[z0_M]])
r_dot_M_init = np.asarray([[vx0_M],[vy0_M],[vz0_M]])  

[A_M_LVLH_init,_,_,_] = M_to_LVLH(r_M_init.reshape(3),r_dot_M_init.reshape(3))
rho_init_LVLH = np.asarray([[rho_x0_LVLH],[rho_y0_LVLH],[rho_z0_LVLH]]).reshape((3,1))
rho_init_M = ((A_M_LVLH_init.T)@rho_init_LVLH).reshape(3)
[rho_x0_M,rho_y0_M,rho_z0_M] = ((A_M_LVLH_init.T)@rho_init_LVLH).reshape(3)

h_init = cross(initial_conditions_M[:3],initial_conditions_M[3:])
der_state_init = propagator_absolute(initial_conditions_M,0,mu)
r_ddot_M_init = der_state_init[3:6]
omega_lm_LVLH_init = np.zeros(3)
omega_lm_LVLH_init[1] = - la.norm(h_init)/(la.norm(initial_conditions_M[:3])**2)
omega_lm_LVLH_init[2] = -la.norm(initial_conditions_M[:3])/(la.norm(h_init)**2) * np.dot(h_init, r_ddot_M_init)

rho_dot_init_LVLH = np.zeros((3,1))
rho_dot_init_LVLH[0] = rho_vx0_LVLH
rho_dot_init_LVLH[1] = rho_vy0_LVLH
rho_dot_init_LVLH[2] = rho_vz0_LVLH

rho_dot_init_M = (A_M_LVLH_init.T)@(rho_dot_init_LVLH + cross(omega_lm_LVLH_init.reshape(3),rho_init_LVLH.reshape(3)).reshape((3,1)))
rho_vx0_M = rho_dot_init_M[0,0]
rho_vy0_M = rho_dot_init_M[1,0]
rho_vz0_M = rho_dot_init_M[2,0]

initial_conditions_chaser_M = [x0_M + rho_x0_M, y0_M + rho_y0_M, z0_M + rho_z0_M, vx0_M + rho_vx0_M, vy0_M + rho_vy0_M, vz0_M + rho_vz0_M]

pos_bary = ((R.T @ np.asarray(initial_conditions_chaser_M[:3]).reshape((3,1))) + np.asarray([[1-mu],[0],[0]]).reshape((3,1))).reshape(3)
vel_bary = (R.T @ np.asarray(initial_conditions_chaser_M[3:6]).reshape((3,1))).reshape((3,))

# Writing the initial conditions in a csv file
# Saving the target and chaser spacecrafts' trajectories in a csv file
file = open("dynamics\initial_conditions.csv","w")
writer = csv.writer(file)

# 1st saving the initial conditions in the Moon frame for the target part and in the LVLH frame for the chaser part
data = np.concatenate((initial_conditions_chaser_LVLH,[period]))
writer.writerow(data)

# 2nd saving the initial conditions in the Moon frame
data = np.concatenate((initial_conditions_M,initial_conditions_chaser_M,[period]))
writer.writerow(data)

# 3rd saving the initial conditions in the barycenter frame
# print(initial_conditions_bary,initial_conditions_chaser_pos_bary[0,:].reshape(3),initial_conditions_chaser_vel_bary[:,0].shape)
data = np.concatenate((initial_conditions_bary,np.asarray([pos_bary[0,0],pos_bary[0,1],pos_bary[0,2],vel_bary[0,0],vel_bary[0,1],vel_bary[0,2]]),[period]))
writer.writerow(data)

# From Murakami's article
# 100km meet-point, in the synodic station-center reference frame, position of the chaser with respect to the target
# CAREFUL the frame is rotated compared to the Moon (synodic one) like the barycenter one is
V2 = (R @ np.asarray([20/r12,70/r12,0]).reshape((3,1))).reshape(3) # rotating it to match the orientation of the Moon frame
# We have no info on the velocity at this meet point, just use the random one
# at the initial conditions we currently have
V2_M = initial_position_M.reshape(3) + V2
V2_M = V2_M.reshape((3,))

# Initial conditions in the Moon frame for both the target and the chaser
data = np.concatenate((initial_conditions_M,[V2_M[0,0],V2_M[0,1],V2_M[0,2]],initial_conditions_chaser_M[3:],[period]))
writer.writerow(data)

V2_LVLH = (A_M_LVLH_init @ V2.reshape((3,1))).reshape(3)
# print(V2_LVLH)
data = np.concatenate((initial_conditions_M,[V2_LVLH[0,0],V2_LVLH[0,1],V2_LVLH[0,2]],initial_conditions_chaser_LVLH[9:],[period]))
writer.writerow(data)

file.close()
