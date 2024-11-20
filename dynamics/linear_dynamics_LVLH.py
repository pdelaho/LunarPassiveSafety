import numpy as np
import numpy.linalg as la


from useful_small_functions import skew


def M_to_LVLH(r, r_dot):
    """Computes the rotation matrix to go from the Moon (synodic) reference frame to the Local-Horizontal-Local-Vertical (LVLH) reference frame

    Args:
        r (3D vector): target's position vector expressed in the Moon frame
        r_dot (3D vector): target's velocity vector expressed in the Moon frame

    Returns:
        3x3 matrix: rotation matrix to go from the Moon to the LVLH frame
    """
    
    r = r.reshape(3)
    r_dot = r_dot.reshape(3)
    h = np.cross(r, r_dot)
    
    # Defining the LVLH reference frame expressed in the Moon synodic frame
    j = - h / la.norm(h)
    k = - r / la.norm(r)
    i = np.cross(j, k)
    
    # Defining the the Moon synodic reference frame
    i_m = np.asarray([1, 0, 0]).reshape(3)
    j_m = np.asarray([0, 1, 0]).reshape(3)
    k_m = np.asarray([0, 0, 1]).reshape(3)
    
    # Computing the rotation matrix to go from the Moon synodic frame to the LVLH frame
    A_M_LVLH = np.zeros((3, 3))
    A_M_LVLH[0, 0] = np.dot(i_m, i)
    A_M_LVLH[0, 1] = np.dot(j_m, i)
    A_M_LVLH[0, 2] = np.dot(k_m, i)
    A_M_LVLH[1, 0] = np.dot(i_m, j)
    A_M_LVLH[1, 1] = np.dot(j_m, j)
    A_M_LVLH[1, 2] = np.dot(k_m, j)
    A_M_LVLH[2, 0] = np.dot(i_m, k)
    A_M_LVLH[2, 1] = np.dot(j_m, k)
    A_M_LVLH[2, 2] = np.dot(k_m, k)
    
    return A_M_LVLH, i, j, k

def propagator_absolute(state, t, mu):
    """Computes the state derivative in the context of the CR3BP using the dynamics of the target as described in Franzini's paper when looking
    at the relative motion

    Args:
        state (6x1 vector): [x,y,z,vx,vy,vz], state vector where we have the position and velocity vectors of the spacecraft expressed in the 
                            Moon frame
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        6x1 vector: derivative of the state vector
    """
    
    ds = np.zeros(6)
    r = np.asarray(state[:3]).reshape((3, 1))
    r_dot = np.asarray(state[3:6]).reshape(3)
    
    # First derivative of the target's position
    ds[:3] = r_dot
    omega_mi = np.asarray([0, 0, 1]).reshape(3) # angular velocity vector of the Moon with respect to the inertial frame in the Moon frame
    r_em = np.asarray([[-1], [0], [0]]).reshape((3, 1)) # position of the Moon with respect to the Earth in the Moon frame
    
    # Second derivative of the target's position
    r_norm = la.norm(r)
    r_es_norm = la.norm(r + r_em)
    r_ddot = - (2 * np.cross(omega_mi, r_dot).reshape((3, 1))) - (np.cross(omega_mi, np.cross(omega_mi, r.reshape(3))).reshape((3, 1))) \
        - (mu * r / (r_norm**3)) - ((1 - mu) * ((r + r_em) / r_es_norm**3 - r_em))
    ds[3:6] = r_ddot.reshape(3)
    
    return ds


def matrix_dynamics(state, mu):
    """Computes the state transition matrix according to the linearized dynamics in Franzini's paper

    Args:
        state (12x1 vector): [x,y,z,vx,vy,vz,rho_x,rho_y,rho_z,rho_dot_x,rho_dot_y,rho_dot_z], state vector where the first 6 parameters describe
                            the motion of the target spacecraft and the last 6 ones describe the relative motion of the chaser
        t (scalar): time step at which we want to compute the state transition matrix
        mu (scalar): mass ratio parameter of the system

    Returns:
        6x6 matrix: state transition matrix
    """
    
    A = np.zeros((6, 6)) # matrix in the state differential equation
    A[:3, 3:] = np.eye(3)
    
    # Define the position and velocity of target in Moon frame
    r_M = np.asarray(state[:3]).reshape((3, 1))
    r_M_norm = la.norm(r_M)
    r_dot_M = np.asarray(state[3:6]).reshape(3)
    
    # Compute angular momentum in Moon frame
    h_M = np.cross(r_M.reshape(3), r_dot_M)
    h_M_norm = la.norm(h_M)
    
    # Rotation matrix from the Moon to the LVLH frames
    [A_M_LVLH, _, j, _] = M_to_LVLH(r_M, r_dot_M)

    # Computing the angular velocity between the Moon synodic and the inertial frame
    ω_mi_M = np.asarray([0, 0, 1]).reshape(3)
    
    # Position vector of the Earth with respect to the Moon
    r_em_M = np.asarray([[-1], [0], [0]]).reshape((3, 1))
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state, 0, mu)
    r_ddot_M = np.asarray(der[3:6]).reshape(3)
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    ω_lm_LVLH = np.zeros((3, 1))
    ω_lm_LVLH[1] = - h_M_norm / r_M_norm**2
    ω_lm_LVLH[2] = - r_M_norm / (h_M_norm**2) * np.dot(h_M, r_ddot_M)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame
    ω_li_LVLH = ω_lm_LVLH + A_M_LVLH @ (ω_mi_M.reshape((3, 1)))
    Ω_li_LVLH = skew(ω_li_LVLH)
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    h_dot = - np.dot(np.cross(r_M.reshape(3), r_ddot_M), j.reshape(3))
    r_dot = (1 / r_M_norm) * np.dot(r_M.reshape(3), r_dot_M)
    
    # Third derivative of the target's position
    r_es_M = r_M + r_em_M
    r_es_M_norm = la.norm(r_es_M)
    r_dddot_M = - (2 * np.cross(ω_mi_M, r_ddot_M).reshape((3, 1))) \
        - (np.cross(ω_mi_M, np.cross(ω_mi_M, r_dot_M)).reshape((3, 1))) \
        - (mu * (1 / r_M_norm**3) * (np.eye(3) - 3 * r_M @ r_M.T / (r_M_norm**2)) @ r_dot_M.reshape((3, 1))) \
        - ((1 - mu) * (1 / r_es_M_norm**3) * (np.eye(3) - 3 * (r_es_M) @ (r_es_M).T / (r_es_M_norm**2)) @ r_dot_M.reshape((3, 1)))
        
    ω_lm_dot_LVLH = np.zeros((3, 1))
    ω_lm_dot_LVLH[1] = - (1 / r_M_norm) * (h_dot / (r_M_norm) + 2 * r_dot * ω_lm_LVLH[1])
    ω_lm_dot_LVLH[2] = (r_dot / r_M_norm - 2 * h_dot / h_M_norm) * ω_lm_LVLH[2] \
        - r_M_norm / (h_M_norm**2) * np.dot(h_M, r_dddot_M.reshape(3))
        
    ω_li_dot_LVLH = ω_lm_dot_LVLH - np.cross(ω_lm_LVLH.reshape(3), (A_M_LVLH @ ω_mi_M.reshape((3, 1))).reshape(3)).reshape((3, 1))
    Ω_li_dot_LVLH = skew(ω_li_dot_LVLH)
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_M_LVLH @ r_es_M
    r_LVLH = A_M_LVLH @ r_M
    A_rho_rho_dot = - (Ω_li_dot_LVLH + Ω_li_LVLH @ Ω_li_LVLH \
        + mu / (r_M_norm**3) * (np.eye(3) - 3 * r_LVLH @ r_LVLH.T / (r_M_norm**2)) \
        + (1 - mu) / (r_es_M_norm**3) * (np.eye(3) - 3 * sum_LVLH @ sum_LVLH.T / (r_es_M_norm**2)))
    
    A[3:,:3] = A_rho_rho_dot
    A[3:,3:] = - 2 * Ω_li_LVLH
    
    return A


def propagator_relative(state, t, mu):
    """Computes the state derivative in the context of the CR3BP using the linear relative dynamics from Franzini's paper

    Args:
        state (12x1 vector): [x,y,z,vx,vy,vz,rho_x,rho_y,rho_z,rho_dot_x,rho_dot_y,rho_dot_z], state vector where the first 6 parameters describe
                            the motion of the target spacecraft and the last 6 ones describe the relative motion of the chaser
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        12x1 vector: derivative of the state vector
    """
    
    ds = np.zeros((12,)) # Derivative of the state vector
    
    # Define the position and velocity of target in Moon frame
    r_M = np.asarray(state[:3]).reshape((3, 1))
    r_M_norm = la.norm(r_M)
    r_dot_M = np.asarray(state[3:6]).reshape(3)
    
    # Compute angular momentum in Moon frame
    h_M = np.cross(r_M.reshape(3), r_dot_M)
    h_M_norm = la.norm(h_M)
    
    # Define the relative position and velocity of chaser in LVLH frame
    ρ_LVLH = np.asarray(state[6:9]).reshape((3, 1))
    ρ_dot_LVLH = np.asarray(state[9:12]).reshape((3, 1))
    
    # Rotation matrix from the Moon to the LVLH frames
    [A_M_LVLH, _, j, _] = M_to_LVLH(r_M, r_dot_M)
     
    # First derivative of the target's position
    ds[:3] = r_dot_M
    
    # Computing the angular velocity between the Moon synodic and the inertial frame
    ω_mi_M = np.asarray([0, 0, 1]).reshape(3)
    
    # Position vector of the Earth with respect to the Moon
    r_em_M = np.asarray([[-1], [0], [0]]).reshape((3, 1))
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state, t, mu)
    r_ddot_M = np.asarray(der[3:6]).reshape(3)
    ds[3:6] = r_ddot_M.reshape(3)
    
    # First derivative of the chaser's relative position
    ds[6:9] = ρ_dot_LVLH.reshape(3)
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    ω_lm_LVLH = np.zeros((3,1))
    ω_lm_LVLH[1] = - h_M_norm / r_M_norm**2
    ω_lm_LVLH[2] = - r_M_norm / (h_M_norm**2) * np.dot(h_M, r_ddot_M)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame
    ω_li_LVLH = ω_lm_LVLH + A_M_LVLH @ (ω_mi_M.reshape((3, 1)))
    Ω_li_LVLH = skew(ω_li_LVLH)
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    h_dot = - np.dot(np.cross(r_M.reshape(3), r_ddot_M), j.reshape(3))
    r_dot = (1 / la.norm(r_M)) * np.dot(r_M.reshape(3), r_dot_M)
    
    # Third derivative of the target's position
    r_es_M = r_M + r_em_M
    r_dddot_M = - (2 * np.cross(ω_mi_M, r_ddot_M).reshape((3, 1))) \
        - (np.cross(ω_mi_M, np.cross(ω_mi_M, r_dot_M)).reshape((3, 1))) \
        - (mu * (1 / r_M_norm**3) * (np.eye(3) - 3 * r_M @ r_M.T / (r_M_norm**2)) @ r_dot_M.reshape((3, 1))) \
        - ((1 - mu) * (1 / la.norm(r_es_M)**3) * (np.eye(3) - 3 * (r_es_M) @ (r_es_M).T / (la.norm(r_es_M)**2)) @ r_dot_M.reshape((3, 1)))
        
    ω_lm_dot_LVLH = np.zeros((3, 1))
    ω_lm_dot_LVLH[1] = - (1 / r_M_norm) * (h_dot / (r_M_norm) + 2 * r_dot * ω_lm_LVLH[1])
    ω_lm_dot_LVLH[2] = (r_dot / r_M_norm - 2 * h_dot / h_M_norm) * ω_lm_LVLH[2] \
        - r_M_norm / (h_M_norm**2) * np.dot(h_M, r_dddot_M.reshape(3))
        
    ω_li_dot_LVLH = ω_lm_dot_LVLH - np.cross(ω_lm_LVLH.reshape(3), (A_M_LVLH @ ω_mi_M.reshape((3, 1))).reshape(3)).reshape((3, 1))
    Ω_li_dot_LVLH = skew(ω_li_dot_LVLH)
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_M_LVLH @ r_es_M
    r_LVLH = A_M_LVLH @ r_M
    rho_ddot = - 2 * Ω_li_LVLH @ ρ_dot_LVLH - (Ω_li_dot_LVLH + Ω_li_LVLH @ Ω_li_LVLH \
        + mu / (r_M_norm**3) * (np.eye(3) - 3 * r_LVLH @ r_LVLH.T / (r_M_norm**2)) \
        + (1 - mu) / (la.norm(r_es_M)**3) * (np.eye(3) - 3 * sum_LVLH @ sum_LVLH.T / (la.norm(r_es_M)**2))) @ ρ_LVLH
    
    ds[9:12] = rho_ddot.reshape(3)
    
    return ds


def integrate_matrix(state, t, mu):
    ds = np.zeros((12,)) # Derivative of the state vector
    ds[:3] = np.asarray(state[3:6]).reshape(3)
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state, t, mu)
    ds[3:6] = np.asarray(der[3:6]).reshape(3)
    
    # Computing the matrix to get the derivative of the relative distance and velocity of the chaser
    A = matrix_dynamics(state, mu)
    der = (A @ state[6:].reshape((6, 1))).reshape(6)
    ds[6:] = der
    
    return ds