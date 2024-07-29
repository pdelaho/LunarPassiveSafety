import numpy as np
import json

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



def load_traj_data(fname):
    """
    Load trajectory data from a json file.
    """

    with open(fname, 'r') as json_file:
        data_dict = json.load(json_file)
        
    t     = np.array(data_dict['t'])
    state = np.array(data_dict['state'])
    mu    = data_dict['mu']
    LU    = data_dict['LU']
    TU    = data_dict['TU']  
    
    return t, state, mu, LU, TU
