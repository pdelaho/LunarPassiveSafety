import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
from scipy.integrate import odeint

from useful_small_functions import *
# from dynamics_linearized import *

def matrix_synodic_to_lvlh(traj):
    r = traj[:3].reshape(3)
    r_dot = traj[3:6].reshape(3)
    
    h = np.cross(r,r_dot)
    
    j_lvlh = - h/la.norm(h)
    k_lvlh = - r/la.norm(r)
    i_lvlh = np.cross(j_lvlh,k_lvlh)
    
    i_syn = np.asarray([1,0,0]).reshape(3)
    j_syn = np.asarray([0,1,0]).reshape(3)
    k_syn = np.asarray([0,0,1]).reshape(3)
    
    A_syn_to_lvlh = np.zeros((3,3))
    A_syn_to_lvlh[0,0] = np.dot(i_syn,i_lvlh)
    A_syn_to_lvlh[0,1] = np.dot(j_syn,i_lvlh)
    A_syn_to_lvlh[0,2] = np.dot(k_syn,i_lvlh)
    A_syn_to_lvlh[1,0] = np.dot(i_syn,j_lvlh)
    A_syn_to_lvlh[1,1] = np.dot(j_syn,j_lvlh)
    A_syn_to_lvlh[1,2] = np.dot(k_syn,j_lvlh)
    A_syn_to_lvlh[2,0] = np.dot(i_syn,k_lvlh)
    A_syn_to_lvlh[2,1] = np.dot(j_syn,k_lvlh)
    A_syn_to_lvlh[2,2] = np.dot(k_syn,k_lvlh)
    
    return A_syn_to_lvlh

def dynamics_synodic(state,t,mu):
    ds = np.zeros(6)
    r = np.asarray(state[:3]).reshape((3,1))
    r_dot = np.asarray(state[3:6]).reshape(3)
    
    ds[:3] = r_dot.reshape(3)
    
    omega_mi = np.asarray([0,0,1]).reshape(3)
    r_em = np.asarray([[-1],[0],[0]]).reshape((3,1))
    
    r_ddot = -2*np.cross(omega_mi,r_dot).reshape((3,1)) - np.cross(omega_mi,np.cross(omega_mi,r.reshape(3))).reshape((3,1)) - mu*r/(np.linalg.norm(r)**3) - (1-mu)*((r+r_em)/(np.linalg.norm(r+r_em)**3) - r_em)
    ds[3:6] = r_ddot.reshape(3)
    
    return ds

def synodic_to_lvlh(syn_traj, target_traj, mu):
    rho_syn = syn_traj[:3] - target_traj[:3]
    rho_dot_syn = syn_traj[3:6] - target_traj[3:6]
    
    A_syn_to_lvlh = matrix_synodic_to_lvlh(target_traj)
    rho_lvlh = A_syn_to_lvlh @ (rho_syn.reshape((3,1)))
    
    h = np.cross(target_traj[:3].reshape(3),target_traj[3:6].reshape(3))
    der = dynamics_synodic(target_traj[:6],0,mu)
    r_ddot = der[3:6]
    omega_lm_lvlh = np.zeros((3,1))
    omega_lm_lvlh[1] = - la.norm(h)/(la.norm(target_traj[:3])**2)
    omega_lm_lvlh[2] = - la.norm(target_traj[:3])/(la.norm(h)**2) * np.dot(h.reshape(3),r_ddot.reshape(3))
    rho_dot_lvlh = A_syn_to_lvlh @ rho_dot_syn.reshape((3,1)) - np.cross(omega_lm_lvlh.reshape(3),rho_lvlh.reshape(3)).reshape((3,1))
    # print(rho_lvlh,rho_dot_lvlh)
    return np.concatenate((rho_lvlh.reshape(3),rho_dot_lvlh.reshape(3)))

def lvlh_to_synodic(lvlh_traj, target_traj, mu):
    r_syn = np.asarray(target_traj[:3])
    r_dot_syn = np.asarray(target_traj[3:6])  

    A_syn_to_lvlh = matrix_synodic_to_lvlh(target_traj[:6])
    rho_lvlh = np.asarray(lvlh_traj[:3]).reshape((3,1))
    [rho_x0_syn, rho_y0_syn, rho_z0_syn] = ((A_syn_to_lvlh.T) @ rho_lvlh).reshape(3)

    h = np.cross(r_syn, r_dot_syn)
    der_state = dynamics_synodic(target_traj[:6],0,mu)
    r_ddot_syn = der_state[3:6]
    omega_lm_lvlh = np.zeros(3)
    omega_lm_lvlh[1] = - la.norm(h)/(la.norm(target_traj[:3])**2)
    omega_lm_lvlh[2] = -la.norm(target_traj[:3])/(la.norm(h)**2) * np.dot(h, r_ddot_syn)

    rho_dot_lvlh = np.zeros((3,1))
    rho_dot_lvlh[0] = lvlh_traj[3]
    rho_dot_lvlh[1] = lvlh_traj[4]
    rho_dot_lvlh[2] = lvlh_traj[5]

    rho_dot_syn = (A_syn_to_lvlh.T)@(rho_dot_lvlh + np.cross(omega_lm_lvlh.reshape(3),rho_lvlh.reshape(3)).reshape((3,1)))
    rho_vx0_syn = rho_dot_syn[0,0]
    rho_vy0_syn = rho_dot_syn[1,0]
    rho_vz0_syn = rho_dot_syn[2,0]

    # initial_conditions_chaser_M = [x0_M + rho_x0_M, y0_M + rho_y0_M, z0_M + rho_z0_M, vx0_M + rho_vx0_M, vy0_M + rho_vy0_M, vz0_M + rho_vz0_M]
    synodic_traj = target_traj[:6].reshape(6) + np.asarray([rho_x0_syn, rho_y0_syn, rho_z0_syn, rho_vx0_syn, rho_vy0_syn, rho_vz0_syn]).reshape(6)
    
    return synodic_traj

def bary_to_synodic(bary_traj, mu):
    R = np.matrix([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]) # Rotation matrix to go from the bary to the synodic frame
    
    syn_pos = R @ (bary_traj[:3].reshape((3,1)) - np.asarray([1-mu, 0, 0]).reshape((3,1)))
    syn_vel = R @ bary_traj[3:6].reshape((3,1))

    return np.concatenate(([syn_pos[i,0] for i in range(3)], [syn_vel[j,0] for j in range(3)]))

def linearized_trans(mu, traj):
    A = np.zeros((6,6)) # matrix in the state differential equation
    A[:3,3:] = np.eye(3)
    
    r_syn = np.asarray(traj[:3]).reshape((3,1))
    r_dot_syn = np.asarray(traj[3:6]).reshape(3)
    h_syn = np.cross(r_syn.reshape(3),r_dot_syn)
    
    A_syn_to_lvlh = matrix_synodic_to_lvlh(traj)
    
    omega_mi_syn = np.asarray([0,0,1]).reshape(3)
    
    # Position vector of the Earth with respect to the Moon
    r_em_syn = np.asarray([[-1],[0],[0]]).reshape((3,1))
    
    der = dynamics_synodic(traj,0,mu)
    r_ddot_syn = np.asarray(der[3:6]).reshape(3)
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    omega_lm_lvlh = np.zeros((3,1))
    omega_lm_lvlh[1] = - la.norm(h_syn)/la.norm(r_syn)**2
    omega_lm_lvlh[2] = - la.norm(r_syn)/(la.norm(h_syn)**2) * np.dot(h_syn,r_ddot_syn)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame
    omega_li_lvlh = omega_lm_lvlh + A_syn_to_lvlh@(omega_mi_syn.reshape((3,1)))
    Omega_li_lvlh = skew(omega_li_lvlh)
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    j_lvlh = -h_syn / la.norm(h_syn)
    h_dot = - np.dot(np.cross(r_syn.reshape(3),r_ddot_syn),j_lvlh.reshape(3))
    r_dot = (1/la.norm(r_syn))*np.dot(r_syn.reshape(3),r_dot_syn)
    
    r_dddot_syn = -2*np.cross(omega_mi_syn,r_ddot_syn).reshape((3,1)) - np.cross(omega_mi_syn,np.cross(omega_mi_syn,r_dot_syn.reshape(3))).reshape((3,1)) \
        - mu*(1/la.norm(r_syn)**3)*(np.eye(3) - 3*r_syn@r_syn.T/(la.norm(r_syn)**2))@r_dot_syn.reshape((3,1)) - (1-mu)*(1/la.norm(r_syn+r_em_syn)**3)*(np.eye(3) \
        - 3*(r_syn+r_em_syn)@(r_syn+r_em_syn).T/(la.norm(r_syn+r_em_syn)**2))@r_dot_syn.reshape((3,1))
    omega_lm_dot_lvlh = np.zeros((3,1))
    omega_lm_dot_lvlh[1] = -(1/la.norm(r_syn))*(h_dot/(la.norm(r_syn)**2) + 2*r_dot*omega_lm_lvlh[1])
    omega_lm_dot_lvlh[2] = (r_dot/la.norm(r_syn) - 2*h_dot/la.norm(h_syn))*omega_lm_lvlh[2] \
        - la.norm(r_syn)/(la.norm(h_syn)**2)*np.dot(h_syn,r_dddot_syn.reshape(3))
    omega_li_dot_lvlh = omega_lm_dot_lvlh - np.cross(omega_lm_lvlh.reshape(3),(A_syn_to_lvlh@omega_mi_syn.reshape((3,1))).reshape(3)).reshape((3,1))
    Omega_li_dot_lvlh = skew(omega_li_dot_lvlh)
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_syn_to_lvlh@(r_syn+r_em_syn)
    r_LVLH = A_syn_to_lvlh@r_syn
    A_rho_rho_dot = - (Omega_li_dot_lvlh + Omega_li_lvlh@Omega_li_lvlh \
        + mu/(la.norm(r_syn)**3) * (np.eye(3) -3*r_LVLH@r_LVLH.T/(la.norm(r_syn)**2)) + (1-mu)/(la.norm(r_syn+r_em_syn)**3) * (np.eye(3) \
        - 3*sum_LVLH@sum_LVLH.T/(la.norm(r_syn+r_em_syn)**2)))
    
    A[3:,:3] = A_rho_rho_dot
    A[3:,3:] = -2*Omega_li_lvlh
    
    return A

def get_traj_ref(initial_conditions_target, M0, horizon, period, mu, n_time):
    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_time-1 != 0:
        dt = horizon*period/(n_time-1)
    else:
        dt = 0.

    time = np.empty(shape=(n_time,), dtype=float)
    traj = np.empty(shape=(n_time, 6), dtype=float)

    time[0] = 0
    t_init = period * np.radians(M0) / (2 * np.pi) - period / 2 # Using the following definition of the mean anomaly: M = 2*pi*t/period
    t_sim = np.linspace(0,t_init,1000)
    traj_prel = integrate.odeint(dynamics_synodic,initial_conditions_target,t_sim,args=(mu,))
    
    for iter in range(n_time-1):
        time[iter+1] = time[iter] + dt
    
    trajectory = integrate.odeint(dynamics_synodic,traj_prel[-1,:],time,args=(mu,))
    
    for iter in range(n_time):
        traj[iter] = trajectory[iter,:]
    
    return traj, time, dt

def get_chaser_nonlin_traj(initial_condition, target_traj, time, mu):
    chaser_traj_lvlh = np.zeros_like(target_traj)
    # 1st step, convert the initial condition from lvlh to synodic
    initial_condition_syn = lvlh_to_synodic(initial_condition, target_traj[0,:], mu)
    
    # 2nd step, integrate the non-linear dynamics, return all of this instead of just the last position/velocity
    chaser_traj_syn = odeint(dynamics_synodic, initial_condition_syn, time, args=(mu,))
    
    # 3rd step, transform the final condition back to the lvlh frame
    # chaser_traj_lvlh = synodic_to_lvlh(chaser_traj_syn[-1,:], target_traj[-1,:], mu)
    for i in range(target_traj.shape[0]):
        chaser_traj_lvlh[i] = synodic_to_lvlh(chaser_traj_syn[i,:], target_traj[i,:], mu)

    return chaser_traj_lvlh