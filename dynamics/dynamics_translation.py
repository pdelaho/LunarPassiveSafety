import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

from useful_small_functions import *

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
    t_init = period * M0 / (2 * np.pi) - period / 2 # Using the following definition of the mean anomaly: M = 2*pi*t/period
    t_sim = np.linspace(0,t_init,1000)
    traj_prel = integrate.odeint(dynamics_synodic,initial_conditions_target,t_sim,args=(mu,))
    
    for iter in range(n_time-1):
        time[iter+1] = time[iter] + dt
    
    trajectory = integrate.odeint(dynamics_synodic,traj_prel[-1,:],time,args=(mu,))
    
    for iter in range(n_time):
        traj[iter] = trajectory[iter,:]
    
    return traj, time, dt