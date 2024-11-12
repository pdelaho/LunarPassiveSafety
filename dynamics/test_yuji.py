import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# from code_yuji import *
from get_initial_conditions import *
from linear_dynamics_LVLH import *

def derivative_lin(state, t, mu):
    # state needs to be size 12 to contain both the target and the chaser's position and velocity
    # the target state is in the synodic frame and the chaser's one is in the LVLH frame
    der = np.zeros((12,))
    r_mi = state[:3]
    r_mi_dot = state[3:6]
    der[:3] = r_mi.reshape((3,))
    
    ω_mi = np.array([ 0, 0, 1])
    r_em = np.array([-1, 0, 0])
    r_ei = r_em + r_mi
    
    # state derivative for the target spacecraft
    der[3:6] = -2 * np.cross(ω_mi, r_mi_dot) - np.cross(ω_mi, np.cross(ω_mi, r_mi)) \
        - mu * r_mi / (np.linalg.norm(r_mi)**3) - ( 1 - mu) * (r_ei / (np.linalg.norm(r_ei)**3) - r_em / (np.linalg.norm(r_em)**3))
    
    # computing the derivative of the chaser's state vector
    # A = dyn_lin_rel_cr3bp_lvlh(state[:6], mu)
    A = matrix_dynamics(state[:6], 0, mu)
    der[6:] = (A @ state[6:].reshape((6,1))).reshape((6,))
    
    return der

def derivative_nonlin(state, t, mu):
    # state is just position and velocity for a single spacecraft this time
    der = np.zeros((6,))
    r_mi = state[:3]
    r_mi_dot = state[3:6]
    
    der[:3] = r_mi.reshape((3,))
    
    ω_mi = np.array([ 0, 0, 1])
    r_em = np.array([-1, 0, 0])
    r_ei = r_em + r_mi
    
    # state derivative for the target spacecraft
    der[3:6] = -2 * np.cross(ω_mi, r_mi_dot) - np.cross(ω_mi, np.cross(ω_mi, r_mi)) \
        - mu * r_mi / (np.linalg.norm(r_mi)**3) - ( 1 - mu) * (r_ei / (np.linalg.norm(r_ei)**3) - r_em / (np.linalg.norm(r_em)**3))
    
    return der
    

def verification(IC_target_syn, IC_chaser_LVLH, IC_chaser_syn, length_sim, TU, LU, mu):
    # this function takes in the initial conditions and returns the error
    # between regular and linearized dynamics
    
    length = length_sim*3600/TU # length_sim is in hour, length is adimensionalized
    t_sim = np.linspace(0,length,1000)
    
    # first computing the chaser's trajectory using non-linear dynamics
    traj_nonlin_syn = odeint(derivative_nonlin, IC_chaser_syn, t_sim, args=(mu,))
    # this trajectory is in the synodic frame
    
    traj_lin_lvlh = odeint(derivative_lin, np.concatenate((IC_target_syn, IC_chaser_LVLH)), t_sim, args=(mu,))
    # target's trajectory is in the synodic frame and chaser's trajetory is in the LVLH frame
    # -> need to get the trajectory of the chaser in the synodic frame
    # traj_nonlin_lvlh = np.empty_like(traj_nonlin_syn)
    error_dist = np.empty((1000,))
    error_vel = np.empty((1000,))
    
    for i in range(1000):
        target_syn = traj_lin_lvlh[i,:6]
        chaser_lvlh = traj_lin_lvlh[i,6:]
        chaser_syn = traj_nonlin_syn[i,:6]
        [rot,_,_,_] = M_to_LVLH(target_syn[:3],target_syn[3:])
        # traj_nonlin_lvlh[i,:3] = ...
        
        rho_syn = chaser_syn[:3] - target_syn[:3]
        rho_dot_syn = chaser_syn[3:6] - target_syn[3:6]
        
        A_M_LVLH,_,_,_ = M_to_LVLH(target_syn[:3],target_syn[3:6])
        rho_LVLH = A_M_LVLH @ (rho_syn.reshape((3,1)))
        
        # compute h with the target's position and velocity in the synodic frame
        h = np.cross(target_syn[:3],target_syn[3:6])
        der = derivative_nonlin(target_syn[:6],0,mu)
        r_ddot = der[3:6]
        omega_lm_LVLH = np.zeros((3,1))
        omega_lm_LVLH[1] = - la.norm(h)/(la.norm(target_syn[:3])**2)
        omega_lm_LVLH[2] = - la.norm(target_syn[:3])/(la.norm(h)**2) * np.dot(h.reshape(3),r_ddot.reshape(3))
        rho_dot_LVLH = A_M_LVLH @ rho_dot_syn.reshape((3,1)) - np.cross(omega_lm_LVLH.reshape(3),rho_LVLH.reshape(3)).reshape((3,1))
        
        error_dist[i] = np.linalg.norm(chaser_lvlh[:3] - rho_LVLH)
        error_vel[i] = np.linalg.norm(chaser_lvlh[3:] - rho_dot_LVLH)
        
    # print(error_dist,error_vel)


LU = 384400 # km, distance between primary attractors
mu = 1.215e-2 # no unit, mass parameter of the system
TU = 1/(2.661699e-6) # s, inverse of the relative angular frequency between the two primary attractors


M = np.linspace(0,360,10)
M_labels = [f"{int(mean_anomaly)}" for mean_anomaly in M ]
base = np.asarray([i for i in range(10,1,-1)])
# rho_init = np.concatenate(([1e-2],1e-2*base, 1e-1*base, base, 10*base))
# rho_init = np.concatenate((10*base, base, 1e-1*base, 1e-2*base, [1e-2]))
rho_init = np.asarray([100,50,10,5,1,0.5,0.1,0.05,0.01])
rho_init_labels = [f"{rho} [km]" for rho in rho_init]


mat_dist = np.empty((len(M),len(rho_init))) # just store the maximum error on the 12 hour propagation
mat_vel = np.empty((len(M),len(rho_init)))


for i in range(len(M)):
    print(i)
    for j in range(len(rho_init)):
        print(j)
        errors_dist = np.empty(5)
        errors_vel = np.empty(5)
        for k in range(5):
            IC_target_M, IC_chaser_LVLH, IC_chaser_M = get_initial_conditions(np.radians(M[i]),rho_init[j],0)
            # now that we have the initial conditions for the target and chaser, we can compute
            verification(IC_target_M, IC_chaser_LVLH, IC_chaser_M, 12, TU, LU, mu)
            # error_distance, error_velocity = verification(IC_target_M, IC_chaser_LVLH, IC_chaser_M)
        #     errors_dist[k] = max(error_distance)
        #     errors_vel[k] = max(error_velocity)
        # mat_dist[i,j] = math.floor(math.log10(np.average(errors_dist)*LU*1e3))
        # mat_vel[i,j] = math.floor(math.log10(np.average(errors_vel)*LU*1e3/TU))

ax2 = sns.heatmap(mat_dist.T, linewidth=0.5)

fig, ax = plt.subplots()
im = ax.imshow(mat_dist.T)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$log_{10}(||\hat{\rho} - \rho||)$", rotation=-90, va="bottom")
ax.set_xticks(np.arange(len(M_labels)), labels=M_labels)
ax.set_yticks(np.arange(len(rho_init_labels)), labels=rho_init_labels)
fig.tight_layout
plt.title(r"Comparing the relative position from linear and non-linear dynamics")
plt.show()

ax2 = sns.heatmap(mat_vel.T, linewidth=0.5)

fig, ax = plt.subplots()
im = ax.imshow(mat_vel.T)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$log_{10}(||\hat{\dot{\rho}} - \dot{\rho}||)$", rotation=-90, va="bottom")
ax.set_xticks(np.arange(len(M_labels)), labels=M_labels)
ax.set_yticks(np.arange(len(rho_init_labels)), labels=rho_init_labels)
fig.tight_layout
plt.title(r"Comparing the relative velocity from linear and non-linear dynamics")
plt.show()