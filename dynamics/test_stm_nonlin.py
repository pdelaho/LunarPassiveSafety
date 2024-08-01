import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la
from numpy import cross
from numpy.random import rand
import matplotlib.pyplot as plt
import csv
import scipy as sc
import os
import sys
import time

from linear_dynamics_LVLH import propagator_absolute,M_to_LVLH,integrate_matrix,matrix_dynamics,propagator_relative
from useful_small_functions import *

r12 = 384400 # km, distance between primary attractors
mu = 1.215e-2 # no unit, mass parameter of the system
TU = 1/(2.661699e-6) # s, inverse of the relative angular frequency between the two primary attractors
L1x = 0.83691513 # nd, position of the L1 point along the x direction
L2x = 1.15568217 # nd, position of the L2 point along the x direction

# Getting the initial conditions from the file
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/initial_conditions.csv"
file = open(fname,"r")
reader = csv.reader(file)
data = []
for lines in reader:
    if lines != []:
        data.append(lines)
file.close()

data = np.asarray(data)
period = data[0,-1]
period = period.astype('float64')
initial_conditions_target_M = np.asarray(data[0,:6])
initial_conditions_target_M = initial_conditions_target_M.astype('float64')
initial_conditions_chaser_LVLH = np.asarray(data[0,6:12])
initial_conditions_chaser_LVLH = initial_conditions_chaser_LVLH.astype('float64')
initial_conditions_chaser_M = np.asarray(data[1,6:12])
initial_conditions_chaser_M = initial_conditions_chaser_M.astype('float64')
initial_conditions_target_bary = np.asarray(data[2,:6])
initial_conditions_target_bary = initial_conditions_target_bary.astype('float64')
initial_conditions_chaser_bary = np.asarray(data[2,6:12])
initial_conditions_chaser_bary = initial_conditions_chaser_bary.astype('float64')
initial_conditions_murakami_M = np.asarray(data[3,:12])
initial_conditions_murakami_M = initial_conditions_murakami_M.astype('float64')
initial_conditions_murakami_LVLH = np.asarray(data[4,:12])
initial_conditions_murakami_LVLH = initial_conditions_murakami_LVLH.astype('float64')

# Setting the time for the simulation
length = 12*3600/TU # nd for a 12-hour simulation
length2 = 100*60/TU
t_simulation = np.linspace(0,period,1000) # change the duration of the simulations HERE

# Simulating the relative (and linear for the chaser) dynamics
# y_orbit = integrate.odeint(propagator_relative,initial_conditions_chaser_LVLH,t_simulation[:],args=(mu,))
# y_orbit = integrate.odeint(propagator_relative,np.concatenate((initial_conditions_target_M,initial_conditions_chaser_LVLH)),t_simulation[:],args=(mu,))
y_orbit = integrate.odeint(propagator_relative,initial_conditions_murakami_LVLH,t_simulation[:],args=(mu,))

# Propagating the non-linear dynamics in the Moon (synodic) frame
# chaser_orbit = integrate.odeint(propagator_absolute,initial_conditions_murakami_M[6:],t_simulation,args=(mu,))
chaser_orbit = integrate.odeint(propagator_absolute,initial_conditions_chaser_M,t_simulation,args=(mu,))

# Plotting the orbit of the target
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(-chaser_orbit[:,0], -chaser_orbit[:,1], chaser_orbit[:,2], color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Moon')
ax.scatter(L2x-(1-mu), 0, 0, label='L2')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Chaser's trajectory in the Moon (synodic) frame")
plt.grid()
plt.show()

# Putting the results from the non-linear dynamics in the LVLH frame
rho_M_history_true = np.zeros_like(chaser_orbit[:,:3])
rho_dot_M_history_true = np.zeros_like(chaser_orbit[:,:3])
rho_LVLH_history_true = np.zeros_like(chaser_orbit[:,:3])
rho_dot_LVLH_history_true = np.zeros_like(chaser_orbit[:,:3])
for i in range(len(chaser_orbit[:,0])):
    rho_M = chaser_orbit[i,:3] - y_orbit[i,:3]
    rho_dot_M = chaser_orbit[i,3:6] - y_orbit[i,3:6]
    
    rho_M_history_true[i] = rho_M.reshape(3)
    rho_dot_M_history_true[i] = rho_dot_M.reshape(3)
    
    A_M_LVLH,_,_,_ = M_to_LVLH(y_orbit[i,:3],y_orbit[i,3:6])
    rho_LVLH = A_M_LVLH @ (rho_M.reshape((3,1)))
    
    h = np.cross(y_orbit[i,:3].reshape(3),y_orbit[i,3:6].reshape(3))
    der = propagator_absolute(y_orbit[i,:6],0,mu)
    r_ddot = der[3:6]
    omega_lm_LVLH = np.zeros((3,1))
    omega_lm_LVLH[1] = - la.norm(h)/(la.norm(y_orbit[i,:3])**2)
    omega_lm_LVLH[2] = - la.norm(y_orbit[i,:3])/(la.norm(h)**2) * np.dot(h.reshape(3),r_ddot.reshape(3))
    rho_dot_LVLH = A_M_LVLH @ rho_dot_M.reshape((3,1)) - np.cross(omega_lm_LVLH.reshape(3),rho_LVLH.reshape(3)).reshape((3,1))
    
    rho_LVLH_history_true[i] = rho_LVLH.reshape(3)
    rho_dot_LVLH_history_true[i] = rho_dot_LVLH.reshape(3)

# Plotting the orbit of the target
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(rho_LVLH_history_true[:,0], rho_LVLH_history_true[:,1], rho_LVLH_history_true[:,2], color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Target')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame")
plt.grid()
plt.show()

# Integrating the linear dynamics again but written in the matrix form in the function
matrix_dyn = integrate.odeint(integrate_matrix,np.concatenate((initial_conditions_target_M,initial_conditions_chaser_LVLH)),t_simulation,args=(mu,))
# matrix_dyn = integrate.odeint(integrate_matrix,np.concatenate((initial_conditions_target_M,initial_conditions_murakami_LVLH[6:])),t_simulation,args=(mu,))

# Plotting the orbit of the target
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(matrix_dyn[:,6], matrix_dyn[:,7], matrix_dyn[:,8], color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Target')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame")
plt.grid()
plt.show()

# Comparing non-linear dynamics and integrating the function with the linear dynamics written in matrix form
error_mat_lin_dist = np.zeros(y_orbit.shape[0])
error_mat_lin_vel = np.zeros(y_orbit.shape[0])
for i in range(y_orbit.shape[0]):
    error = la.norm(matrix_dyn[i,6:9]-rho_LVLH_history_true[i,:3])
    error_mat_lin_dist[i] = error
    error_vel = la.norm(matrix_dyn[i,9:12]-rho_dot_LVLH_history_true[i,:3])
    error_mat_lin_vel[i] = error_vel

plt.plot(t_simulation*TU/3600,error_mat_lin_dist*r12*1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\hat{\rho}_{STM} - \rho_{non-lin}$|| [m]")
plt.title('Error on the relative distance between propagation with STM and non-linear dynamics')
plt.show()

plt.plot(t_simulation*TU/3600,error_mat_lin_vel*r12*1e3/TU)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\dot{\hat{\rho}}_{STM} - \dot{\rho}_{non-lin}$|| [m/s]")
plt.title('Error on the relative velocity between propagation with STM and non-linear dynamics')
plt.show()

# Computing the trajectory multiplying by the STM to propagate and comparing it to non-linear dynamics
chaser_trajectory_stm = np.zeros_like(y_orbit[:,:6])
chaser_trajectory_stm[0] = (np.asarray(initial_conditions_chaser_LVLH)).reshape(6)
# chaser_trajectory_stm[0] = (np.asarray(initial_conditions_murakami_LVLH[6:12])).reshape(6)
timing_exp = np.zeros_like(y_orbit[:,0])
timing_approx = np.zeros_like(y_orbit[:,0])
norm_diff_matrix = np.zeros_like(y_orbit[:,0])
for i in range(1,y_orbit.shape[0]):
    delta_t = t_simulation[i]-t_simulation[i-1] # jouer sur les indices MAIS EN FIXANT LES CONDITIONS INITIALES
    start = time.perf_counter()
    phi = sc.linalg.expm(delta_t*matrix_dynamics(y_orbit[i,:],t_simulation[i],mu)) # jouer sur les indices
    end = time.perf_counter()
    timing_exp[i] = end-start
    start2 = time.perf_counter()
    phi2 = get_phi(delta_t,matrix_dynamics(y_orbit[i,:],t_simulation[i],mu),p=5) # similar with get_phi and sc.linalg.expm
    end2 = time.perf_counter()
    timing_approx[i] = end2 - start2
    norm_diff_matrix[i] = np.trace((phi-phi2) @ ((phi-phi2).T)) # this is the norm squared but will work
    chaser_trajectory_stm[i] = (phi @ chaser_trajectory_stm[i-1].reshape((6,1))).reshape(6)
    # chaser_trajectory_stm[i] = (phi2 @ chaser_trajectory_stm[i-1].reshape((6,1))).reshape(6)

error_stm_lin_dist = np.zeros(y_orbit.shape[0])
error_stm_lin_vel = np.zeros(y_orbit.shape[0])
for i in range(y_orbit.shape[0]):
    error = la.norm(chaser_trajectory_stm[i,:3] - y_orbit[i,6:9])
    error_stm_lin_dist[i] = error
    error_vel = la.norm(chaser_trajectory_stm[i,3:6] - y_orbit[i,9:12])
    error_stm_lin_vel[i] = error_vel
    
plt.plot(t_simulation*TU/3600,error_stm_lin_dist*r12*1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\rho_{STM} - \rho_{int}$|| [m]")
plt.title('Error between integration of linear dynamics and multiplication by STM on relative distance')
plt.show()

plt.plot(t_simulation*TU/3600,error_stm_lin_vel*r12*1e3/TU)
plt.xlabel('Time [hours]')
plt.ylabel(r"||$\dot{\rho}_{STM} - \dot{\rho}_{int}$|| [m/s]")
plt.title('Error between integration of linear dynamics and multiplication by STM on relative velocity')
plt.show()

# Sanity check that initial conditions matches when they should
# print(rho_LVLH_history_true[0,:],rho_x0_LVLH,rho_y0_LVLH,rho_z0_LVLH)
print(rho_LVLH_history_true[0,:],initial_conditions_chaser_LVLH[:3])
# print(rho_LVLH_history_true[0,:],initial_conditions_murakami_LVLH[6:9])

# print(rho_dot_LVLH_history_true[0,:],rho_vx0_LVLH,rho_vy0_LVLH,rho_vz0_LVLH)
print(rho_dot_LVLH_history_true[0,:],initial_conditions_chaser_LVLH[3:])
# print(rho_dot_LVLH_history_true[0,:],initial_conditions_murakami_LVLH[9:12])

# There might still be a mistake in the way I put the results from the Moon frame back in the LVLH one, see plots are kinda different

# Comparing the linear and non-linear dynamics again
error_lin_nonlin = np.zeros(y_orbit.shape[0])
error_lin_nonlin_vel = np.zeros(y_orbit.shape[0])
for i in range(y_orbit.shape[0]):
    error_lin_nonlin[i] = la.norm(y_orbit[i,6:9] - rho_LVLH_history_true[i,:])
    error_lin_nonlin_vel[i] = la.norm(y_orbit[i,9:12] - rho_dot_LVLH_history_true[i,:])

plt.plot(t_simulation*TU/3600,error_lin_nonlin*r12*1e3)
plt.show()

plt.plot(t_simulation*TU/3600,error_lin_nonlin_vel*r12*1e3/TU)
plt.show()

plt.plot(t_simulation*TU/3600,chaser_trajectory_stm[:,:3]*r12*1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r"$\hat{\rho}$ [m]")
plt.title(r'$\rho$ given by multiplying with the STM')
plt.show()

plt.plot(t_simulation*TU/3600,chaser_trajectory_stm[:,3:6]*r12*1e3/TU)
plt.xlabel('Time [hours]')
plt.ylabel(r"$\hat{\dot{\rho}}$ [m/s]")
plt.title(r'$\dot{\rho}$ given by multiplying with the STM')
plt.show()

plt.plot(t_simulation*TU/3600,timing_exp*1e3,label='using scipy')
plt.plot(t_simulation*TU/3600,timing_approx*1e3,label='using approx')
plt.plot(t_simulation*TU/3600,np.average(timing_exp)*np.ones_like(y_orbit[:,0])*1e3,label='mean using scipy')
plt.plot(t_simulation*TU/3600,np.average(timing_approx)*np.ones_like(y_orbit[:,0])*1e3,label='mean using approx')
plt.xlabel('Time [hours]')
plt.ylabel(r"Computation time [ms]")
plt.title('Computation time to get the STM with 2 different methods')
plt.legend()
plt.show()

plt.plot(t_simulation*TU/3600,norm_diff_matrix)
plt.xlabel('Time [hours]')
plt.ylabel(r"Norm of the matrix difference between the STMs")
plt.title('Comparison of the two ways of computing the STM')
plt.show()
