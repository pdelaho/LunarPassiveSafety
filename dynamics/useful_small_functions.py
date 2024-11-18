import numpy as np
import json
import math


def skew(a):
    """Compute the skew matrix associated to vector a

    Args:
        a (3x1 vector): vector for which we want to get the associated skew matrix 

    Returns:
        3x3 matrix: associated skew matrix
    """
    
    mat = np.zeros((3,3))
    mat[0,1] = -a[2]
    mat[0,2] = a[1]
    mat[1,0] = a[2]
    mat[1,2] = -a[0]
    mat[2,0] = -a[1]
    mat[2,1] = a[0]
    
    return mat
    

def get_phi(t, A, p=5):
    """Taken from Yuji's github.
    Numerically computing an approximation of the matrix exp(A*t).

    Args:
        t (float): timestep at which we want to compute exp(A*t)
        A (3x3 matrix): matrix for which we want to compute exp(A*t)
        p (integer): order of the approximation
        
    Returns:
        3x3 matrix: result of exp(A*t)
    """

    phi = np.eye(A.shape[0])
    for i in range(1, p):
        phi += np.linalg.matrix_power(A*t, i) / math.factorial(i)
    return phi


def bary_to_synodic(bary_traj, mu):
    """Takes in a state vector in the barycenter frame and the mass ratio, returns the state
    vector in the synodic frame.

    Args:
        bary_traj (6x1 vector): state vector (position,velocity) in barycenter frame
        mu (float): mass ratio parameter in 3-body problem
        
    Returns:
        6x1 vector: state vector in the synodic frame
    """
    
    R = np.matrix([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]) # Rotation matrix to go from the bary to the synodic frame
    
    syn_pos = R @ (bary_traj[:3].reshape((3,1)) - np.asarray([1-mu, 0, 0]).reshape((3,1)))
    syn_vel = R @ bary_traj[3:6].reshape((3,1))

    return np.concatenate(([syn_pos[i,0] for i in range(3)], [syn_vel[j,0] for j in range(3)]))


def load_traj_data(fname):
    """
    Load trajectory data from a json file.
    """

    with open(fname, 'r') as json_file:
        data_dict = json.load(json_file)
        
    t     = np.array(data_dict['t'])
    state = np.array(data_dict['state']) # in the barycenter frame
    mu    = data_dict['mu']
    LU    = data_dict['LU']
    TU    = data_dict['TU']

    state_syn = np.empty_like(state)
    for i in range(state.shape[0]):
        state_syn[i] = bary_to_synodic(state[i],mu)

    return t, state_syn, mu, LU, TU


def load_traj_data_old(fname):
    """
    Load trajectory data from a json file.
    """

    with open(fname, 'r') as json_file:
        data_dict = json.load(json_file)
        
    t     = np.array(data_dict['t'])
    state = np.array(data_dict['state']) # in the barycenter frame
    mu    = data_dict['mu']
    LU    = data_dict['LU']
    TU    = data_dict['TU']
   
    return t, state, mu, LU, TU


def generate_outside_ellipsoid(inv_P, center):
    """Takes in the inverse of the shape matrix of an ellipsoid and its center, returns a random point
    located outside the ellipsoid.

    Args:
        inv_P (6x6 matrix): inverse of the ellipsoid's shape matrix
        center (6x1 vector): center of the ellipsoid

    Returns:
        6x1 vector: random generated point located outside the ellipsoid
    """
    
    x = np.random.randn(6)*1e-4
    if (x-center) @ inv_P @ (x-center).T <=1:
        generate_outside_ellipsoid(inv_P, center)
    else:
        return x

  
def generate_inside_ellipsoid(inv_P, center):
    """Takes in the inverse of the shape matrix of an ellipsoid and its center, returns a random point
    located inside the ellipsoid.

    Args:
        inv_P (6x6 matrix): inverse of the ellipsoid's shape matrix
        center (6x1 vector): center of the ellipsoid

    Returns:
        6x1 vector: random generated point located inside the ellipsoid
    """
    x = np.random.randn(6)*1e-5 # try 1e-5 if this doesn't work
    if (x-center) @ inv_P @ (x-center).T <= 1:
        return x
    else:
        generate_inside_ellipsoid(inv_P, center)

    
def volume_ellipsoid(shape_matrix, LU, TU, dim=6, type='pos'):
    """Takes in the shape matrix of an ellipsoid, the length and time parameters in the 3-body,
    the dimension of the ellipsoid, and if the ellipsoid is for position or velocity purposes.
    Returns the volume of the ellipsoid for the position or velocity part.

    Args:
        shape_matrix (6x6 matrix): shape matrix of the ellipsoid
        LU (float): length parameter in 3-body problem
        TU (float): time parameter in 3-body problem
        dim (int, optional): dimension of the ellipsoid. Defaults to 6.
        type (str, optional): 'pos' or 'vel' to chose which part of the ellipsoid we want to look at. Defaults to 'pos'.

    Returns:
        float: volume of the position or velocity part of the ellipsoid
    """
    
    eigenvalues, _ = np.linalg.eig(shape_matrix)
    prod = 1
    for i in range(dim):
        prod *= 1/np.sqrt(eigenvalues[i])
        
    if dim == 3:
        if type == 'pos':
            volume = 4/3 * np.pi * prod * LU**3
        if type == 'vel':
            volume = 4/3 * np.pi * prod * LU**3 / (TU**3)
    if dim == 6:
        volume = 1/(np.sqrt(np.pi*dim)) * (2*math.e*np.pi/dim)**(dim/2) * prod * LU**6 / (TU**3)
    
    return volume