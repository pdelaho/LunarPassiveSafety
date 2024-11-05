import numpy as np

from dynamics_translation import *
from useful_small_functions import *


def linearize_translation(mu, traj, time, control):
    """Takes in the mass ratio parameter, the trajectory of the target in the synodic frame, the time steps
    at which we have the state of the target, and if we want to take into account control inputs.

    Args:
        mu (float): mass ratio parameter in 3-body problem
        traj (6xN): target's trajectory in synodic frame at each timestep
        time (Nx1): timesteps at which we have the target's state vector
        control (bool): true if we want to take into account control inputs

    Returns:
        dictionary: regroups the state transition matrices "stm" (N-1x6x6), the control input matrices "cim" (N-1x6x3),
                    and the rotation matrices from synodic to LVLH "psi" (Nx3x3)
    """
    n_time = len(time) 

    psi = np.empty(shape=(n_time, 3, 3), dtype=float) 
    stm = np.empty(shape=(n_time-1, 6, 6), dtype=float)
    cim = np.empty(shape=(n_time-1, 6, 3), dtype=float)
    
    for i in range(n_time):
        
        psi[i] = matrix_synodic_to_lvlh(traj[i])
                    
        if i < n_time-1:
            delta_t = time[i+1] - time[i]
            stm[i] = get_phi(delta_t,linearized_trans(mu, traj[i,:]))
            
            if control:
                cim[i] = np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)
            else:
                cim[i] = np.zeros((6,3))
        
    mats = {"stm": stm, "cim": cim, "psi": psi}

    return mats


# TO DO: change the name of this function to linearize_translation_BRS
def linearize_translation_scvx(mu, traj, time, control):
    """Takes in the mass ratio parameter, the target's trajectory in the synodic frame, the time steps at which
    we have the target's state vector, and if we want to take into account control inputs.

    Args:
        mu (float): mass ratio parameter in 3-body problem
        traj (6xN): target's trajectory in synodic frame at each timestep
        time (Nx1): timesteps at which we have the target's state vector (simulation time + backward reachable sets timesteps)
        control (bool): true if we want to take into account control inputs

    Returns:
        dictionary: regroups the state transition matrices "stm" (N-1x6x6), the control input matrices "cim" (N-1x6x3),
                    and the rotation matrices from synodic to LVLH "psi" (Nx3x3)
    """
    n_time = len(time)
    n_stm = len(traj) # n_stm = n_time + N_BRS

    # TO DO: stop using n_time, do it all with n_stm because we might need rotation and control matrices
    # after the end (and code gets easier to read) -> wouldn't that just be the same as the function above?
    # -> be careful when using that function
    psi = np.empty(shape=(n_time, 3, 3), dtype=float) 
    stm = np.empty(shape=(n_stm-1, 6, 6), dtype=float)
    cim = np.empty(shape=(n_time-1, 6, 3), dtype=float)
    
    for i in range(n_time):
        
        psi[i] = matrix_synodic_to_lvlh(traj[i])
                    
        if i < n_time-1:
            delta_t = time[i+1] - time[i]
            stm[i] = get_phi(delta_t,linearized_trans(mu, traj[i,:]))
            cim[i] = np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)
    
    for i in range(n_stm):
        if i < n_stm - 1:
            delta_t = time[i+1] - time[i]
            stm[i] = get_phi(delta_t,linearized_trans(mu, traj[i,:]))
        
    mats = {"stm": stm, "cim": cim, "psi": psi}

    return mats