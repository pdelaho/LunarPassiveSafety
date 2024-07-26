import numpy as np

from dynamics_translation import *
from useful_small_functions import *
from tqdm import tqdm


def linearize_translation(mu, traj, time, control):
    n_time = len(time) 
    dt = time[1]-time[0]

    psi      = np.empty(shape=(n_time, 3, 3), dtype=float)   # synodic -> LVLH rotation matrix 
    stm      = np.empty(shape=(n_time-1, 6, 6),   dtype=float)
    cim      = np.empty(shape=(n_time-1, 6, 3),   dtype=float)
    
    # for i in tqdm(range(n_time)):
    for i in range(n_time):
        
        psi[i] = matrix_synodic_to_lvlh(traj[i])
                    
        if i < n_time-1:
            delta_t = time[i+1] - time[i] # if equally spaced time steps, this is the same as dt
            stm[i] = get_phi(delta_t,linearized_trans(mu, traj[i,:]))
            
            if control:
                cim[i] = np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0) # constant in my implementation
            else:
                cim[i] = np.zeros((6,3))
        
    mats = {"stm": stm, "cim": cim, "psi": psi}

    return mats

