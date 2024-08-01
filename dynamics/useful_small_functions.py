import numpy as np
import json
import math

def skew(a):
    """Compute the skew matrix associated to vector a

    Args:
        a (3x1 vector): vector from which we wanna compute the skew matrix
    """
    mat = np.zeros((3,3))
    mat[0,1] = -a[2]
    mat[0,2] = a[1]
    mat[1,0] = a[2]
    mat[1,2] = -a[0]
    mat[2,0] = -a[1]
    mat[2,1] = a[0]
    return mat

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

def bary_to_synodic(bary_traj, mu):
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
    x = np.random.randn(6)*1e-4
    if (x-center) @ inv_P @ (x-center).T <=1:
        generate_outside_ellipsoid(inv_P, center)
    else:
        return x
    
def generate_inside_ellipsoid(inv_P, center):
    x = np.random.randn(6)*1e-5 # try 1e-5 if this doesn't work
    if (x-center) @ inv_P @ (x-center).T <= 1:
        return x
    else:
        generate_inside_ellipsoid(inv_P, center)
        
def volume_ellipsoid(shape_matrix, LU, TU, dim=6, type='pos'):
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
