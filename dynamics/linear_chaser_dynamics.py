# WRONG FILE, SEE THE FILE linear_dynamics_LVLH.py FOR THE MOST UP-TO-DATE VERSION OF THE FUNCTIONS

import numpy as np
import numpy.linalg as la 
import scipy.integrate as integrate
import scipy as sc

def M_to_LVLH(r,r_dot):
    """Computes the rotation matrix to go from the Moon (synodic) reference frame to the Local-Horizontal-Local-Vertical (LVLH) reference frame

    Args:
        r (3D vector): target's position vector expressed in the Moon frame
        r_dot (3D vector): target's velocity vector expressed in the Moon frame

    Returns:
        3x3 matrix: rotation matrix to go from the Moon to the LVLH frame
    """
    r = r.reshape(3)
    r_dot = r_dot.reshape(3)
    h = np.cross(r,r_dot)
    
    # Defining the LVLH reference frame expressed in the Moon synodic frame
    j = - h/la.norm(h)
    k = - r/la.norm(r)
    i = np.cross(j,k)
    # print(la.norm(i))
    
    # Defining the the Moon synodic reference frame
    i_m = np.zeros(3)
    i_m[0] = 1
    j_m = np.zeros(3)
    j_m[1] = 1
    k_m = np.zeros(3)
    k_m[2] = 1
    
    # Computing the rotation matrix to go from the Moon synodic frame to the LVLH frame
    A_M_LVLH = np.zeros((3,3))
    A_M_LVLH[0,0] = np.dot(i_m,i)
    A_M_LVLH[0,1] = np.dot(j_m,i)
    A_M_LVLH[0,2] = np.dot(k_m,i)
    A_M_LVLH[1,0] = np.dot(i_m,j)
    A_M_LVLH[1,1] = np.dot(j_m,j)
    A_M_LVLH[1,2] = np.dot(k_m,j)
    A_M_LVLH[2,0] = np.dot(i_m,k)
    A_M_LVLH[2,1] = np.dot(j_m,k)
    A_M_LVLH[2,2] = np.dot(k_m,k)
    
    return A_M_LVLH,i,j,k

def propagator_relative(state,t,mu):
    # Maybe try to separate some parts of this functions into subfunctions that I could reuse in the stm function for instance
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
    r_M = np.zeros((3,1)) # Position vector of the target expressed in the Moon frame
    r_M[0] = state[0]
    r_M[1] = state[1]
    r_M[2] = state[2]
    r_dot_M = np.zeros(3) # Velocity of the target expressed in the Moon frame
    r_dot_M[0] = state[3]
    r_dot_M[1] = state[4]
    r_dot_M[2] = state[5]
    h_M = np.cross(r_M.reshape(3),r_dot_M) # Angular momentum vector expressed in the Moon frame
    rho_LVLH = np.zeros((3,1)) # Relative position of the chaser with respect to the target in the LVLH frame
    rho_LVLH[0] = state[6]
    rho_LVLH[1] = state[7]
    rho_LVLH[2] = state[8]
    rho_dot_LVLH = np.zeros((3,1)) # Relative velocity of the chaser with respect to the target in the LVLH frame
    rho_dot_LVLH[0] = state[9]
    rho_dot_LVLH[1] = state[10]
    rho_dot_LVLH[2] = state[11]
    
    # Getting the rotation matrix from the Moon to the LVLH frames
    [A_M_LVLH,_,j,_] = M_to_LVLH(r_M,r_dot_M)
     
    # First derivative of the target's position
    ds[0] = r_dot_M[0]
    ds[1] = r_dot_M[1]
    ds[2] = r_dot_M[2]
    
    # Computing the angular velocity between the Moon synodic and the inertial frame
    omega_mi_M = np.zeros(3)
    omega_mi_M[2] = 1 # since it is adimensionalized, in the moon frame
    
    # Position vector of the Earth with respect to the Moon
    r_em_M = np.zeros((3,1))
    r_em_M[0] = -1 # adimensionalized still, in the moon frame
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state,t,mu)
    r_ddot_M = np.zeros(3)
    r_ddot_M[0] = der[3]
    r_ddot_M[1] = der[4]
    r_ddot_M[2] = der[5]
    ds[3] = r_ddot_M[0]
    ds[4] = r_ddot_M[1]
    ds[5] = r_ddot_M[2]
    
    # First derivative of the chaser's relative position
    ds[6] = rho_dot_LVLH[0]
    ds[7] = rho_dot_LVLH[1]
    ds[8] = rho_dot_LVLH[2]
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    omega_lm_LVLH = np.zeros((3,1))
    omega_lm_LVLH[1] = - la.norm(h_M)/la.norm(r_M)**2
    omega_lm_LVLH[2] = - la.norm(r_M)/(la.norm(h_M)**2) * np.dot(h_M,r_ddot_M)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame
    omega_li_LVLH = omega_lm_LVLH + A_M_LVLH@(omega_mi_M.reshape((3,1)))
    Omega_li_LVLH = np.zeros((3,3))
    Omega_li_LVLH[0,1] = -omega_li_LVLH[2]
    Omega_li_LVLH[0,2] = omega_li_LVLH[1]
    Omega_li_LVLH[1,0] = omega_li_LVLH[2]
    Omega_li_LVLH[1,2] = -omega_li_LVLH[0]
    Omega_li_LVLH[2,0] = -omega_li_LVLH[1]
    Omega_li_LVLH[2,1] = omega_li_LVLH[0]
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    h_dot = - np.dot(np.cross(r_M.reshape(3),r_ddot_M),j.reshape(3))
    r_dot = (1/la.norm(r_M))*np.dot(r_M.reshape(3),r_dot_M)
    
    # Third derivative of the target's position
    r_dddot_M = -2*np.cross(omega_mi_M,r_ddot_M).reshape((3,1)) - np.cross(omega_mi_M,np.cross(omega_mi_M,r_dot_M.reshape(3))).reshape((3,1)) \
        - mu*(1/la.norm(r_M)**3)*(np.eye(3) - 3*r_M@r_M.T/(la.norm(r_M)**2))@r_dot_M.reshape((3,1)) - (1-mu)*(1/la.norm(r_M+r_em_M)**3)*(np.eye(3) \
        - 3*(r_M+r_em_M)@(r_M+r_em_M).T/(la.norm(r_M+r_em_M)**2))@r_dot_M.reshape((3,1))
    omega_lm_dot_LVLH = np.zeros((3,1))
    omega_lm_dot_LVLH[1] = -(1/la.norm(r_M))*(h_dot/(la.norm(r_M)**2) + 2*r_dot*omega_lm_LVLH[1])
    omega_lm_dot_LVLH[2] = (r_dot/la.norm(r_M) - 2*h_dot/la.norm(h_M))*omega_lm_LVLH[2] \
        - la.norm(r_M)/(la.norm(h_M)**2)*np.dot(h_M,r_dddot_M.reshape(3))
    omega_li_dot_LVLH = omega_lm_dot_LVLH - np.cross(omega_lm_LVLH.reshape(3),(A_M_LVLH@omega_mi_M.reshape((3,1))).reshape(3)).reshape((3,1))
    
    Omega_li_dot_LVLH = np.zeros((3,3))
    Omega_li_dot_LVLH[0,1] = -omega_li_dot_LVLH[2]
    Omega_li_dot_LVLH[0,2] = omega_li_dot_LVLH[1]
    Omega_li_dot_LVLH[1,0] = omega_li_dot_LVLH[2]
    Omega_li_dot_LVLH[1,2] = -omega_li_dot_LVLH[0]
    Omega_li_dot_LVLH[2,0] = -omega_li_dot_LVLH[1]
    Omega_li_dot_LVLH[2,1] = omega_li_dot_LVLH[0]
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_M_LVLH@(r_M+r_em_M)
    r_LVLH = A_M_LVLH@r_M
    rho_ddot = -2*Omega_li_LVLH@rho_dot_LVLH - (Omega_li_dot_LVLH + Omega_li_LVLH@Omega_li_LVLH \
        + mu/(la.norm(r_M)**3) * (np.eye(3) -3*r_LVLH@r_LVLH.T/(la.norm(r_M)**2)) + (1-mu)/(la.norm(r_M+r_em_M)**3) * (np.eye(3) \
        - 3*sum_LVLH@sum_LVLH.T/(la.norm(r_M+r_em_M)**2)))@rho_LVLH
    
    ds[9] = rho_ddot[0]
    ds[10] = rho_ddot[1]
    ds[11] = rho_ddot[2]
    
    return ds

def propagator_relative_with_matrix(state,delta_t,mu,t=0):
    # Maybe try to separate some parts of this functions into subfunctions that I could reuse in the stm function for instance
    """Computes the state derivative in the context of the CR3BP using the linear relative dynamics from Franzini's paper

    Args:
        state (12x1 vector): [x,y,z,vx,vy,vz,rho_x,rho_y,rho_z,rho_dot_x,rho_dot_y,rho_dot_z], state vector where the first 6 parameters describe
                            the motion of the target spacecraft and the last 6 ones describe the relative motion of the chaser
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        12x1 vector: derivative of the state vector
    """
    matrix = np.zeros((6,6)) # Matrix A of the linearized dynamics
    r_M = np.zeros((3,1)) # Position vector of the target expressed in the Moon frame
    r_M[0] = state[0]
    r_M[1] = state[1]
    r_M[2] = state[2]
    r_dot_M = np.zeros(3) # Velocity of the target expressed in the Moon frame
    r_dot_M[0] = state[3]
    r_dot_M[1] = state[4]
    r_dot_M[2] = state[5]
    h_M = np.cross(r_M.reshape(3),r_dot_M) # Angular momentum vector expressed in the Moon frame
    
    # Getting the rotation matrix from the Moon to the LVLH frames
    [A_M_LVLH,_,j,_] = M_to_LVLH(r_M,r_dot_M)
     
    # Computing the angular velocity between the Moon synodic and the inertial frame
    omega_mi_M = np.zeros(3)
    omega_mi_M[2] = 1 # since it is adimensionalized, in the moon frame
    
    # Position vector of the Earth with respect to the Moon
    r_em_M = np.zeros((3,1))
    r_em_M[0] = -1 # adimensionalized still, in the moon frame
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state,t,mu)
    r_ddot_M = np.zeros(3)
    r_ddot_M[0] = der[3]
    r_ddot_M[1] = der[4]
    r_ddot_M[2] = der[5]
    
    # Angular velocity and skew-matrix between the LVLH and the Moon frame expressed in the LVLH frame
    omega_lm_LVLH = np.zeros((3,1))
    omega_lm_LVLH[1] = - la.norm(h_M)/la.norm(r_M)**2
    omega_lm_LVLH[2] = - la.norm(r_M)/(la.norm(h_M)**2) * np.dot(h_M,r_ddot_M)
    
    # Angular velocity between the LVLH and the inertial frame expressed in the LVLH frame -> could be a skew function
    omega_li_LVLH = omega_lm_LVLH + A_M_LVLH@(omega_mi_M.reshape((3,1)))
    Omega_li_LVLH = np.zeros((3,3))
    Omega_li_LVLH[0,1] = -omega_li_LVLH[2]
    Omega_li_LVLH[0,2] = omega_li_LVLH[1]
    Omega_li_LVLH[1,0] = omega_li_LVLH[2]
    Omega_li_LVLH[1,2] = -omega_li_LVLH[0]
    Omega_li_LVLH[2,0] = -omega_li_LVLH[1]
    Omega_li_LVLH[2,1] = omega_li_LVLH[0]
    
    # Derivatives of the norms of the angular momentum and the target's position's vector
    h_dot = - np.dot(np.cross(r_M.reshape(3),r_ddot_M),j.reshape(3))
    r_dot = (1/la.norm(r_M))*np.dot(r_M.reshape(3),r_dot_M)
    
    # Third derivative of the target's position
    r_dddot_M = -2*np.cross(omega_mi_M,r_ddot_M).reshape((3,1)) - np.cross(omega_mi_M,np.cross(omega_mi_M,r_dot_M.reshape(3))).reshape((3,1)) \
        - mu*(1/la.norm(r_M)**3)*(np.eye(3) - 3*r_M@r_M.T/(la.norm(r_M)**2))@r_dot_M.reshape((3,1)) - (1-mu)*(1/la.norm(r_M+r_em_M)**3)*(np.eye(3) \
        - 3*(r_M+r_em_M)@(r_M+r_em_M).T/(la.norm(r_M+r_em_M)**2))@r_dot_M.reshape((3,1))
    omega_lm_dot_LVLH = np.zeros((3,1))
    omega_lm_dot_LVLH[1] = -(1/la.norm(r_M))*(h_dot/(la.norm(r_M)**2) + 2*r_dot*omega_lm_LVLH[1])
    omega_lm_dot_LVLH[2] = (r_dot/la.norm(r_M) - 2*h_dot/la.norm(h_M))*omega_lm_LVLH[2] \
        - la.norm(r_M)/(la.norm(h_M)**2)*np.dot(h_M,r_dddot_M.reshape(3))
    omega_li_dot_LVLH = omega_lm_dot_LVLH - np.cross(omega_lm_LVLH.reshape(3),(A_M_LVLH@omega_mi_M.reshape((3,1))).reshape(3)).reshape((3,1))
    
    Omega_li_dot_LVLH = np.zeros((3,3))
    Omega_li_dot_LVLH[0,1] = -omega_li_dot_LVLH[2]
    Omega_li_dot_LVLH[0,2] = omega_li_dot_LVLH[1]
    Omega_li_dot_LVLH[1,0] = omega_li_dot_LVLH[2]
    Omega_li_dot_LVLH[1,2] = -omega_li_dot_LVLH[0]
    Omega_li_dot_LVLH[2,0] = -omega_li_dot_LVLH[1]
    Omega_li_dot_LVLH[2,1] = omega_li_dot_LVLH[0]
    
    # Second derivative of the chaser's relative position
    sum_LVLH = A_M_LVLH@(r_M+r_em_M)
    r_LVLH = A_M_LVLH@r_M
    A_rho_rho_dot = - (Omega_li_dot_LVLH + Omega_li_LVLH@Omega_li_LVLH \
        + mu/(la.norm(r_M)**3) * (np.eye(3) -3*r_LVLH@r_LVLH.T/(la.norm(r_M)**2)) + (1-mu)/(la.norm(r_M+r_em_M)**3) * (np.eye(3) \
        - 3*sum_LVLH@sum_LVLH.T/(la.norm(r_M+r_em_M)**2)))
    matrix[:3,3:] = np.eye(3)
    matrix[3:,3:] = -2*Omega_li_dot_LVLH
    matrix[3:,:3] = A_rho_rho_dot
    
    # print(delta_t*matrix)
    stm = sc.linalg.expm(delta_t*matrix)
    return matrix

def propagator_absolute(state,t,mu):
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
    r = np.zeros((3,1))
    r[0] = state[0]
    r[1] = state[1]
    r[2] = state[2]
    r_dot = np.zeros(3)
    r_dot[0] = state[3]
    r_dot[1] = state[4]
    r_dot[2] = state[5]
    
    # First derivative of the target's position
    ds[0] = r_dot[0]
    ds[1] = r_dot[1]
    ds[2] = r_dot[2]
    
    # Second derivative of the target's position
    omega_mi = np.zeros(3)
    omega_mi[2] = 1 # since it is adimensionalized, in the moon frame
    r_em = np.zeros((3,1))
    r_em[0] = -1 # adimensionalized still, in the moon frame
    # Computing the second derivative with respect to the moon frame
    r_ddot = -2*np.cross(omega_mi,r_dot).reshape((3,1)) - np.cross(omega_mi,np.cross(omega_mi,r.reshape(3))).reshape((3,1)) - mu*r/(np.linalg.norm(r)**3) - (1-mu)*((r+r_em)/(np.linalg.norm(r+r_em)**3) - r_em)
    ds[3] = r_ddot[0]
    ds[4] = r_ddot[1]
    ds[5] = r_ddot[2]
    return ds

# From Yuji's github code
def get_phi(t, A, p=5):
    """
        numerically computing the matrix exp(A*t)
        p: order of the approximation
    """
    phi = np.eye(A.shape[0])
    for i in range(1, p):
        phi += np.linalg.matrix_power(A*t, i) / np.math.factorial(i)
    return phi

def propagator_absolute2(state,t,mu):
    """Computes the state derivative in the context of the CR3BP using the dynamics as described in the beginning of Franzini's paper

    Args:
        state (6x1 vector): [x,y,z,vx,vy,vz], state vector where we have the position and velocity vectors of the spacecraft expressed in the 
                            Moon frame
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        6x1 vector: derivative of the state vector
    """
    ds = np.zeros(6)
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    r_mi = np.sqrt(x**2 + y**2 + z**2)
    r_ei = np.sqrt((x-1)**2 + y**2 + z**2)
    
    # First derivative of the position in the Moon frame
    ds[0] = vx
    ds[1] = vy
    ds[2] = vz
    
    # Second derivative of the position in the Moon frame
    ds[3] = 2*vy + x - mu*x/(r_mi**3) - (1-mu)*((x-1)/(r_ei**3) + 1)
    ds[4] = -2*vx + y - mu*y/(r_mi**3) - (1-mu)*y/(r_ei**3)
    ds[5] = -mu*z/(r_mi**3) - (1-mu)*z/(r_ei**3)
    
    return ds

def halo_propagator(state,t,mu):
    """Computes the state derivative in the context of the CR3BP using the dynamics as described in Shane Ross's textbook about the 3-body problem

    Args:
        state (6x1 vector): [x,y,z,vx,vy,vz], state vector where we have the position and velocity vectors of the spacecraft expressed in the 
                            barycenter reference frame
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        6x1 vector: derivative of the state vector
    """
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    
    ds = np.zeros(6)
    ds[0] = vx
    ds[1] = vy
    ds[2] = state[5]
    ds[3] = x + 2*vy - (1-mu)*(x+mu)/(r1**3) - mu*(x-1+mu)/(r2**3)
    ds[4] = y - 2*vx -(1-mu)*y/(r1**3) - mu*y/(r2**3)
    ds[5] = -(1-mu)*z/(r1**3) - mu*z/(r2**3)
    return ds

def halo_propagator_with_STM(state,t,mu):
    """Computes the state derivative in the context of the CR3BP using the dynamics as described in Shane Ross's textbook about the 3-body problem
    considering the state transition matrix in the state as well

    Args:
        state (42x1 vector): [x,y,z,vx,vy,vz,phi11,phi12,phi13,phi14,phi15,phi16,phi21,phi22,phi23,phi24,phi25,phi26,phi31,phi32,phi33,phi34,phi35,
                            phi36,phi41,phi42,phi43,phi44,phi45,phi46,phi51,phi52,phi53,phi54,phi55,phi56,phi61,phi62,phi63,phi64,phi65,phi66], 
                            state vector where we have the position and velocity vectors of the spacecraft expressed in the barycenter reference 
                            frame and the coefficients of the 6x6 state transition matrix phi
        t (scalar): time step at which we want to compute the derivative
        mu (scalar): mass ratio parameter of the system

    Returns:
        42x1 vector: derivative of the state vector
    """
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    
    ds = np.zeros((42,1))
    ds[0] = vx
    ds[1] = vy
    ds[2] = vz
    ds[3] = x + 2*vy - (1-mu)*(x+mu)/(r1**3) - mu*(x-1+mu)/(r2**3)
    ds[4] = y - 2*vx -(1-mu)*y/(r1**3) - mu*y/(r2**3)
    ds[5] = -(1-mu)*z/(r1**3) - mu*z/(r2**3)
    
    dUdxx = 1 - (1-mu)/(r1**3) + 3*(1-mu)*(x+mu)**2/(r1**5) - mu/(r2**3) + 3*mu*(x-1+mu)**2/(r2**5)
    dUdyy = 1 - (1-mu)/(r1**3) + 3*(1-mu)*(y**2)/(r1**5) - mu/(r2**3) + 3*mu*(y**2)/(r2**5)
    dUdzz = -(1-mu)/(r1**3) + 3*(1-mu)*(z**2)/(r1**5) - mu/(r2**3) + 3*mu*(z**2)/(r2**5)
    dUdxy = 3*(1-mu)*(x+mu)*y/(r1**5) + 3*mu*(x-1+mu)*y/(r2**5)
    dUdxz = 3*(1-mu)*(x+mu)*z/(r1**5) + 3*mu*(x-1+mu)*z/(r2**5)
    dUdyz = 3*(1-mu)*y*z/(r1**5) + 3*mu*y*z/(r2**5)
    
    A = np.matrix([[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [dUdxx, dUdxy, dUdxz, 0, 2, 0],
         [dUdxy, dUdyy, dUdyz, -2, 0, 0],
         [dUdxz, dUdyz, dUdzz, 0, 0, 0]])

    STM = np.matrix([[state[6], state[7], state[8], state[9], state[10], state[11]],
           [state[12], state[13], state[14], state[15], state[16], state[17]],
           [state[18], state[19], state[20], state[21], state[22], state[23]],
           [state[24], state[25], state[26], state[27], state[28], state[29]],
           [state[30], state[31], state[32], state[33], state[34], state[35]],
           [state[36], state[37], state[38], state[39], state[40], state[41]]])
    
    dSTMdt = A@STM
    ds[6:12] = dSTMdt.T[:,0]
    ds[12:18] = dSTMdt.T[:,1]
    ds[18:24] = dSTMdt.T[:,2]
    ds[24:30] = dSTMdt.T[:,3]
    ds[30:36] = dSTMdt.T[:,4]
    ds[36:42] = dSTMdt.T[:,5]
    
    return ds.reshape(42)

def single_shooting(initial_state,residual,jacobian):
    """Computes the new initial conditions of the orbit given the previous initial conditions, the residuals after half of the orbit, and the 
    jacobian matrix

    Args:
        initial_state (3x1 vector): previous initial state, containing the x component of the position vector, the y component of the velocity
                                    vector, and the length of half an orbit
        residual (3x1 vector): residuals after propagating the orbit for half a period
        jacobian (3x3 matrix): jacobian matrix of the function describing the dynamics

    Returns:
        3x1 vector: new (and should be more accurate) initial conditions for a given orbit
    """
    new_initial_state = initial_state.reshape((3,1)) - np.linalg.pinv(jacobian)@(residual.reshape((3,1)))
    
    return new_initial_state

def integrate_matrix1(state,t,mu):
    ds = np.zeros((12,)) # Derivative of the state vector
    r_M = np.zeros((3,1)) # Position vector of the target expressed in the Moon frame
    r_M[0] = state[0]
    r_M[1] = state[1]
    r_M[2] = state[2]
    r_dot_M = np.zeros(3) # Velocity of the target expressed in the Moon frame
    r_dot_M[0] = state[3]
    r_dot_M[1] = state[4]
    r_dot_M[2] = state[5]
    rho_LVLH = np.zeros((3,1)) # Relative position of the chaser with respect to the target in the LVLH frame
    rho_LVLH[0] = state[6]
    rho_LVLH[1] = state[7]
    rho_LVLH[2] = state[8]
    rho_dot_LVLH = np.zeros((3,1)) # Relative velocity of the chaser with respect to the target in the LVLH frame
    rho_dot_LVLH[0] = state[9]
    rho_dot_LVLH[1] = state[10]
    rho_dot_LVLH[2] = state[11]
         
    # First derivative of the target's position
    ds[0] = r_dot_M[0]
    ds[1] = r_dot_M[1]
    ds[2] = r_dot_M[2]
    
    # Computing the angular velocity between the Moon synodic and the inertial frame
    omega_mi_M = np.zeros(3)
    omega_mi_M[2] = 1 # since it is adimensionalized, in the moon frame
    
    # Position vector of the Earth with respect to the Moon
    r_em_M = np.zeros((3,1))
    r_em_M[0] = -1 # adimensionalized still, in the moon frame
    
    # Second derivative of the target's position expressed in the Moon frame
    der = propagator_absolute(state,t,mu)
    r_ddot_M = np.zeros(3)
    r_ddot_M[0] = der[3]
    r_ddot_M[1] = der[4]
    r_ddot_M[2] = der[5]
    ds[3] = r_ddot_M[0]
    ds[4] = r_ddot_M[1]
    ds[5] = r_ddot_M[2]
    
    A = propagator_relative_with_matrix(state,0,mu,t)
    der = A @ state[6:].reshape((6,1))
    ds[6] = der[0]
    ds[7] = der[1]
    ds[8] = der[2]
    ds[9] = der[3]
    ds[10] = der[4]
    ds[11] = der[5]
    
    return ds
    