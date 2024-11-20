import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
from scipy.integrate import odeint


from useful_small_functions import skew


def matrix_synodic_to_lvlh(traj):
    """Takes it the state vector (position,velocity) of a spacecraft
    in the synodic frame (centered on the Moon) and returns the
    rotation matrix to go from the synodic to the local-vertical-local-
    horizontal (LVLH) frame.

    Args:
        traj (6x1 vector): state vector of the spacecraft (position,velocity)

    Returns:
        3x3 matrix: rotation matrix to go from the synodic to the LVLH frames
    """
    
    r = traj[:3].reshape(3)
    r_dot = traj[3:6].reshape(3)
    
    h = np.cross(r, r_dot)
    
    j_lvlh = - h / la.norm(h)
    k_lvlh = - r / la.norm(r)
    i_lvlh = np.cross(j_lvlh, k_lvlh)
    
    i_syn = np.asarray([1, 0, 0]).reshape(3)
    j_syn = np.asarray([0, 1, 0]).reshape(3)
    k_syn = np.asarray([0, 0, 1]).reshape(3)
    
    A_syn_to_lvlh = np.zeros((3, 3))
    A_syn_to_lvlh[0,0] = np.dot(i_syn, i_lvlh)
    A_syn_to_lvlh[0,1] = np.dot(j_syn, i_lvlh)
    A_syn_to_lvlh[0,2] = np.dot(k_syn, i_lvlh)
    A_syn_to_lvlh[1,0] = np.dot(i_syn, j_lvlh)
    A_syn_to_lvlh[1,1] = np.dot(j_syn, j_lvlh)
    A_syn_to_lvlh[1,2] = np.dot(k_syn, j_lvlh)
    A_syn_to_lvlh[2,0] = np.dot(i_syn, k_lvlh)
    A_syn_to_lvlh[2,1] = np.dot(j_syn, k_lvlh)
    A_syn_to_lvlh[2,2] = np.dot(k_syn, k_lvlh)
    
    return A_syn_to_lvlh


def dynamics_synodic(state, t, mu):
    """Takes in a state vector in the synodic frame and returns 
    the derivative using the dynamics in the synodic frame. It has 
    to take the time to be able to use odeint on that function. It 
    also needs the parameter mu to be able to compute the dynamics.

    Args:
        state (6x1 vector): state vector (position,velocity)
        t (float): time step at which we have the corresponding state vector
        mu (float): mass ratio as defined in the 3-body problem

    Returns:
        6x1 vector: derivative of state vector in the synodic frame
    """
    
    ds = np.zeros(6) # Initialize the derivative vector
    
    # Extract position and velocity
    r = np.asarray(state[:3]).reshape((3, 1))
    r_dot = np.asarray(state[3:6]).reshape(3)
    
    # Update the derivative of the position (velocity)
    ds[:3] = r_dot.reshape(3)
    
    # Constants
    ω_mi = np.asarray([0, 0, 1]).reshape(3)
    r_em = np.asarray([[-1], [0], [0]]).reshape((3, 1))
    
    r_norm = np.linalg.norm(r)
    r_em_sum = r + r_em
    r_em_sum_norm = np.linalg.norm(r_em_sum)
    
    # Compute the second derivative of position (acceleration)
    r_ddot = - 2 * np.cross(ω_mi, r_dot).reshape((3, 1)) \
             - np.cross(ω_mi, np.cross(ω_mi, r.reshape(3))).reshape((3, 1)) \
             - mu * r / (r_norm**3) \
             - (1 - mu) * (r_em_sum / (r_em_sum_norm**3) - r_em)
             
    ds[3:6] = r_ddot.reshape(3)
    
    return ds


def synodic_to_lvlh(syn_traj, target_traj, mu):
    """Takes in the state vector of chaser spacecraft relative to the 
    target in the synodic frame, the target's state vector and the mass 
    ratio parameter mu, and returns the state vector of the chaser in 
    the LVLH frame.

    Args:
        syn_traj (6x1 vector): chaser's state vector (position,velocity) in synodic frame
        target_traj (6x1 vector): target's state vector (position,velocity) in synodic frame
        mu (float): mass ratio parameter in 3-body problem

    Returns:
        6x1 vector: chaser's state vector in the LVLH frame
    """
    
    # Computing relative position and velocity of the chaser in the synodic frame
    ρ_syn = syn_traj[:3] - target_traj[:3]
    ρ_dot_syn = syn_traj[3:6] - target_traj[3:6]
    
    # Rotating the relative position from synodic to LVLH frames
    A_syn_to_lvlh = matrix_synodic_to_lvlh(target_traj)
    ρ_lvlh = A_syn_to_lvlh @ (ρ_syn.reshape((3, 1)))
    
    # Computing momentum and position's second derivative
    h = np.cross(target_traj[:3].reshape(3), target_traj[3:6].reshape(3))
    der = dynamics_synodic(target_traj[:6], 0, mu)
    r_ddot = der[3:6]
    
    # Precomputing the norms
    h_norm = la.norm(h)
    norm_target_traj = la.norm(target_traj[:3])
    
    # Computing the angular velocity vector
    ω_lm_lvlh = np.zeros((3, 1))
    ω_lm_lvlh[1] = - h_norm / (norm_target_traj**2)
    ω_lm_lvlh[2] = - norm_target_traj / (h_norm**2) * np.dot(h.reshape(3), r_ddot.reshape(3))
    
    # Computing the relative velocity in the LVLH frame
    rho_dot_lvlh = A_syn_to_lvlh @ ρ_dot_syn.reshape((3, 1)) \
                   - np.cross(ω_lm_lvlh.reshape(3), ρ_lvlh.reshape(3)).reshape((3, 1))

    return np.concatenate((ρ_lvlh.reshape(3), rho_dot_lvlh.reshape(3)))


def lvlh_to_synodic(lvlh_traj, target_traj, mu):
    """Takes in the state vector of chaser spacecraft relative to the 
    target in the LVLH frame, the target's state vector in the synodic 
    frame and the mass ratio parameter mu, and returns the state vector 
    of the chaser in the synodic frame.

    Args:
        lvlh_traj (6x1 vector): chaser's state vector (position,velocity) in LVLH frame
        target_traj (6x1 vector): target's state vector (position,velocity) in the synodic frame
        mu (float): mass ratio parameter in 3-body problem

    Returns:
        6x1 vector: chaser's state vector in synodic frame
    """
    
    # Extract position and velocity of the target in the synodic frame
    r_syn = np.asarray(target_traj[:3])
    r_dot_syn = np.asarray(target_traj[3:6])  

    # Rotating the relative position of the chaser from LVLH to synodic frames
    A_syn_to_lvlh = matrix_synodic_to_lvlh(target_traj[:6])
    ρ_lvlh = np.asarray(lvlh_traj[:3]).reshape((3, 1))
    [ρ_x0_syn, ρ_y0_syn, ρ_z0_syn] = ((A_syn_to_lvlh.T) @ ρ_lvlh).reshape(3)

    # Computing the momentum and position's second derivative
    h = np.cross(r_syn, r_dot_syn)
    der_state = dynamics_synodic(target_traj[:6], 0, mu)
    r_ddot_syn = der_state[3:6]
    
    # Precomputing the norms
    h_norm = la.norm(h)
    norm_target_traj = la.norm(target_traj[:3])
    
    # Computing the angular velocity vector
    ω_lm_lvlh = np.zeros(3)
    ω_lm_lvlh[1] = - h_norm / (norm_target_traj**2)
    ω_lm_lvlh[2] = - norm_target_traj / (h_norm**2) * np.dot(h, r_ddot_syn)

    # Extracting the relative velocity in the LVLH frame
    ρ_dot_lvlh = np.zeros((3, 1))
    ρ_dot_lvlh = lvlh_traj[3:6].reshape((3, 1))
    # rho_dot_lvlh[0] = lvlh_traj[3]
    # rho_dot_lvlh[1] = lvlh_traj[4]
    # rho_dot_lvlh[2] = lvlh_traj[5]
    
    # Computing the relative velocity in the synodic frame
    ρ_dot_syn = (A_syn_to_lvlh.T) @ (ρ_dot_lvlh + np.cross(ω_lm_lvlh.reshape(3), ρ_lvlh.reshape(3)).reshape((3, 1)))
    ρ_vx0_syn, ρ_vy0_syn, ρ_vz0_syn = ρ_dot_syn[:3, 0]
    # rho_vx0_syn = rho_dot_syn[0,0]
    # rho_vy0_syn = rho_dot_syn[1,0]
    # rho_vz0_syn = rho_dot_syn[2,0]

    # Computing the position of the chaser in the synodic frame
    synodic_traj = target_traj[:6].reshape(6) + np.asarray([ρ_x0_syn, ρ_y0_syn, ρ_z0_syn, ρ_vx0_syn, ρ_vy0_syn, ρ_vz0_syn]).reshape(6)
    
    return synodic_traj


def bary_to_synodic(bary_traj, mu):
    """Takes in a state vector in the barycenter frame and the mass ratio,
    returns the state vector in the synodic frame.

    Args:
        bary_traj (6x1 vector): state vector (position,velocity) in barycenter frame
        mu (float): mass ratio parameter in 3-body problem

    Returns:
        6x1 vector: state vector in the synodic frame
    """
    
    R = np.matrix([[-1, 0, 0], 
                   [0, -1, 0], 
                   [0,  0, 1]]) # Rotation matrix to go from the bary to the synodic frame
    
    syn_pos = R @ (bary_traj[:3].reshape((3, 1)) - np.asarray([1 - mu, 0, 0]).reshape((3, 1)))
    syn_vel = R @ bary_traj[3:6].reshape((3, 1))

    return np.concatenate(([syn_pos[i, 0] for i in range(3)], [syn_vel[j, 0] for j in range(3)]))


def linearized_trans(mu, traj):
    """Takes in the mass ratio parameter and the state vector of the target
    spacecraft in the synodic frame, returns the transition matrix for 
    the chaser's spacecraft in the LVLH frame.

    Args:
        mu (float): mass ratio parameter
        traj (6x1 vector): target's state vector (position,velocity) in synodic frame

    Returns:
        3x3 matrix: state transition matrix for the chaser in LVLH frame
    """
    
    A = np.zeros((6, 6)) # matrix in the state differential equation
    A[:3,3:] = np.eye(3)
    
    # Extracting position and velocity
    r_syn = np.asarray(traj[:3]).reshape((3, 1))
    norm_r_syn = la.norm(r_syn)
    r_dot_syn = np.asarray(traj[3:6]).reshape(3)
    
    # Computing the momentum
    h_syn = np.cross(r_syn.reshape(3), r_dot_syn)
    norm_h_syn = la.norm(h_syn)
    
    # Computing the rotation matrix between synodic and LVLH frames
    A_syn_to_lvlh = matrix_synodic_to_lvlh(traj)
    
    # Computing the angular velocity vector
    ω_mi_syn = np.asarray([0, 0, 1]).reshape(3)
    
    # Position vector of the Earth with respect to the Moon
    r_em_syn = np.asarray([[-1], [0], [0]]).reshape((3, 1))
    r_em_sum_syn = r_syn + r_em_syn
    
    # Computing position's second derivative
    der = dynamics_synodic(traj, 0, mu)
    r_ddot_syn = np.asarray(der[3:6]).reshape(3)
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    ω_lm_lvlh = np.zeros((3, 1))
    ω_lm_lvlh[1] = - norm_h_syn / norm_r_syn**2
    ω_lm_lvlh[2] = - norm_r_syn / (norm_h_syn**2) * np.dot(h_syn, r_ddot_syn)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame
    ω_li_lvlh = ω_lm_lvlh + A_syn_to_lvlh@(ω_mi_syn.reshape((3, 1)))
    Ω_li_lvlh = skew(ω_li_lvlh)
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    j_lvlh = - h_syn / norm_h_syn
    h_dot = - np.dot(np.cross(r_syn.reshape(3), r_ddot_syn), j_lvlh.reshape(3))
    r_dot = (1 / norm_r_syn) * np.dot(r_syn.reshape(3), r_dot_syn)
    
    r_dddot_syn = - 2 * np.cross(ω_mi_syn, r_ddot_syn).reshape((3, 1)) \
                  - np.cross(ω_mi_syn, np.cross(ω_mi_syn, r_dot_syn.reshape(3))).reshape((3, 1)) \
                  - mu * (1 / norm_r_syn**3) * (np.eye(3) - 3 * r_syn @ r_syn.T / (norm_r_syn**2)) @ r_dot_syn.reshape((3, 1)) \
                  - (1 - mu) * (1 / la.norm(r_em_sum_syn)**3) * (np.eye(3) \
        - 3 * (r_em_sum_syn) @ (r_em_sum_syn).T / (la.norm(r_em_sum_syn)**2)) @ r_dot_syn.reshape((3, 1))
    
    omega_lm_dot_lvlh = np.zeros((3, 1))
    # Might be a mistake in here
    omega_lm_dot_lvlh[1] = - (1 / norm_r_syn) * (h_dot / (norm_r_syn) + 2 * r_dot * ω_lm_lvlh[1])
    omega_lm_dot_lvlh[2] = (r_dot / norm_r_syn - 2 * h_dot / norm_h_syn) * ω_lm_lvlh[2] \
        - norm_r_syn / (norm_h_syn**2) * np.dot(h_syn, r_dddot_syn.reshape(3))
    omega_li_dot_lvlh = omega_lm_dot_lvlh - np.cross(ω_lm_lvlh.reshape(3), (A_syn_to_lvlh @ ω_mi_syn.reshape((3, 1))).reshape(3)).reshape((3, 1))
    Omega_li_dot_lvlh = skew(omega_li_dot_lvlh)
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_syn_to_lvlh @ r_em_sum_syn
    r_LVLH = A_syn_to_lvlh @ r_syn
    A_rho_rho_dot = - (Omega_li_dot_lvlh + Ω_li_lvlh @ Ω_li_lvlh \
        + mu / (norm_r_syn**3) * (np.eye(3) - 3 * r_LVLH @ r_LVLH.T / (norm_r_syn**2)) \
        + (1 - mu) / (la.norm(r_em_sum_syn)**3) * (np.eye(3) \
        - 3 * sum_LVLH @ sum_LVLH.T / (la.norm(r_em_sum_syn)**2)))
    
    A[3:,:3] = A_rho_rho_dot
    A[3:,3:] = - 2 * Ω_li_lvlh
    
    return A


def get_traj_ref(initial_conditions_target, M0, horizon, period, mu, n_time):
    # Time discretization (given the number of samples defined in rpod_scenario)
    if n_time - 1 != 0:
        dt = horizon * period / (n_time - 1)
    else:
        dt = 0.

    time = np.empty(shape=(n_time,), dtype=float)
    traj = np.empty(shape=(n_time, 6), dtype=float)

    time[0] = 0
    t_init = period * M0 / (2 * np.pi) - period / 2 # Using the following definition of the mean anomaly: M = 2*pi*t/period
    t_sim = np.linspace(0,t_init, 1000)
    traj_prel = integrate.odeint(dynamics_synodic, initial_conditions_target, t_sim, args=(mu,))
    
    for iter in range(n_time - 1):
        time[iter + 1] = time[iter] + dt
    
    trajectory = integrate.odeint(dynamics_synodic, traj_prel[-1, :], time, args=(mu,))
    
    for iter in range(n_time):
        traj[iter] = trajectory[iter, :]
    
    return traj, time, dt


def get_chaser_nonlin_traj(initial_condition, target_traj, time, mu):
    chaser_traj_lvlh = np.zeros_like(target_traj)
    # 1st step, convert the initial condition from lvlh to synodic
    initial_condition_syn = lvlh_to_synodic(initial_condition, target_traj[0, :], mu)
    
    # 2nd step, integrate the non-linear dynamics, return all of this instead of just the last position/velocity
    chaser_traj_syn = odeint(dynamics_synodic, initial_condition_syn, time, args=(mu,))
    
    # 3rd step, transform the final condition back to the lvlh frame
    # chaser_traj_lvlh = synodic_to_lvlh(chaser_traj_syn[-1,:], target_traj[-1,:], mu)
    for i in range(target_traj.shape[0]):
        chaser_traj_lvlh[i] = synodic_to_lvlh(chaser_traj_syn[i, :], target_traj[i, :], mu)

    return chaser_traj_lvlh


def dynamics_synodic_control(state, t, mu, a):
    ds = np.zeros(6)
    r = np.asarray(state[:3]).reshape((3, 1))
    r_dot = np.asarray(state[3:6]).reshape(3)
    
    ds[:3] = r_dot.reshape(3)
    
    omega_mi = np.asarray([0, 0, 1]).reshape(3)
    r_em = np.asarray([[-1], [0], [0]]).reshape((3, 1))
    
    r_ddot = - 2 * np.cross(omega_mi, r_dot).reshape((3, 1)) - np.cross(omega_mi, np.cross(omega_mi, r.reshape(3))).reshape((3, 1)) \
        - mu * r / (np.linalg.norm(r)**3) - (1 - mu) * ((r + r_em) / (np.linalg.norm(r + r_em)**3) - r_em) + a.reshape((3, 1))
    ds[3:6] = r_ddot.reshape(3)
    
    return ds