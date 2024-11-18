""" This file implements the dynamics of a spacecraft in the context of the Circular Restricted 3-Body Problem with the evolution
of the state transition matrix (STM), designed to be used with odeint """


import numpy as np


def halo_propagator_with_STM(state,t,mu):
    """Dynamics of the Circular Restricted 3 Body Problem in the synodic frame centered on the barycenter of the system including the
    evolution of the State Transition Matrix (STM)

    Args:
        state (42x1 vector): state of the differential equation (3 for position, 3 for velocity, 6x6 for STM)
        t (scalar): time

    Returns:
        42x1 vector: derivative of the state vector according to dynamics
    """
    
    x, y, z, vx, vy, vz = state[:6]

    y2, z2 = y**2, z**2
    r1 = np.sqrt((x + mu)**2 + y2 + z2)
    r2 = np.sqrt((x - 1 + mu)**2 + y2 + z2)
    
    statedot = np.zeros((42,1))
    statedot[:3] = [[vx], [vy], [vz]]
    
    r1_cubed, r2_cubed = r1**3, r2**3
    one_minus_mu = 1 - mu
    statedot[3] = x + 2 * vy - (one_minus_mu * (x + mu) / r1_cubed) - (mu * (x - one_minus_mu) / r2_cubed)
    statedot[4] = y - 2 * vx - (one_minus_mu * y / r1_cubed) - (mu * y / r2_cubed)
    statedot[5] = - one_minus_mu * z / r1_cubed - mu * z / r2_cubed
    
    r1_fifth, r2_fifth = r1**5, r2**5
    dUdxx = 1 - (one_minus_mu / r1_cubed) + (3 * one_minus_mu * (x + mu)**2 / r1_fifth) \
            - (mu / r2_cubed) + (3 * mu * (x - one_minus_mu)**2 / r2_fifth)
            
    dUdyy = 1 - (one_minus_mu / r1_cubed) + (3 * one_minus_mu * (y**2) / r1_fifth) \
            - (mu / r2_cubed) + (3 * mu * (y**2) / r2_fifth)
            
    dUdzz = - (one_minus_mu / r1_cubed) + (3 * one_minus_mu * (z**2) / r1_fifth) \
            - (mu / r2_cubed) + (3 * mu * (z**2) / r2_fifth)
            
    dUdxy = (3 * one_minus_mu * (x + mu) * y / r1_fifth) + (3 * mu * (x - one_minus_mu) * y / r2_fifth)
    
    dUdxz = (3 * one_minus_mu * (x + mu) * z / r1_fifth) + (3 * mu * (x - one_minus_mu) * z / r2_fifth)
    
    dUdyz = (3 * one_minus_mu * y * z / r1_fifth) + (3 * mu * y * z / r2_fifth)
    
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [dUdxx, dUdxy, dUdxz, 0, 2, 0],
        [dUdxy, dUdyy, dUdyz, -2, 0, 0],
        [dUdxz, dUdyz, dUdzz, 0, 0, 0]
        ])

    STM = np.array(state[6:42]).reshape((6, 6))
    dSTMdt = A @ STM
    statedot[6:42] = dSTMdt.T.reshape(-1, 1)

    return statedot.reshape(len(statedot))

def halo_propagator(state,t,mu):
    """Dynamics of the Circular Restricted 3 Body Problem in the synodic frame centered on the barycenter

    Args:
        state (6x1 vector): state of the differential equation (3 for position, 3 for velocity)
        t (scalar): time

    Returns:
        6x1 vector: derivative of the state vector according to dynamics
    """
    
    x, y, z, vx, vy, vz = state
    
    y2, z2 = y**2, z**2
    r1 = np.sqrt((x + mu)**2 + y2 + z2)
    r2 = np.sqrt((x - 1 + mu)**2 + y2 + z2)
    
    statedot = np.zeros((6,1))
    statedot[:3] = vx, vy, vz
    
    r1_cubed, r2_cubed = r1**3, r2**3
    one_minus_mu = 1 - mu
    statedot[3] = x + 2 * vy - (one_minus_mu * (x + mu) / r1_cubed) - (mu * (x - one_minus_mu) / r2_cubed)
    statedot[4] = y - 2 * vx - (one_minus_mu * y / r1_cubed) - (mu * y / r2_cubed)
    statedot[5] = - one_minus_mu * z / r1_cubed - mu * z / r2_cubed
    
    return statedot