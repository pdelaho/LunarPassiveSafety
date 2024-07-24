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
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    
    statedot = np.zeros((42,1)) # Could it be just 42 as a 1D array?
    statedot[0] = vx
    statedot[1] = vy
    statedot[2] = vz
    statedot[3] = x + 2*vy - (1-mu)*(x+mu)/(r1**3) - mu*(x-1+mu)/(r2**3)
    statedot[4] = y - 2*vx -(1-mu)*y/(r1**3) - mu*y/(r2**3)
    statedot[5] = -(1-mu)*z/(r1**3) - mu*z/(r2**3)
    
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
    statedot[6:12] = dSTMdt.T[:,0]
    statedot[12:18] = dSTMdt.T[:,1]
    statedot[18:24] = dSTMdt.T[:,2]
    statedot[24:30] = dSTMdt.T[:,3]
    statedot[30:36] = dSTMdt.T[:,4]
    statedot[36:42] = dSTMdt.T[:,5]
    
    return statedot.reshape(len(statedot))

def halo_propagator(state,t,mu):
    """Dynamics of the Circular Restricted 3 Body Problem in the synodic frame centered on the barycenter

    Args:
        state (6x1 vector): state of the differential equation (3 for position, 3 for velocity)
        t (scalar): time

    Returns:
        42x1 vector: derivative of the state vector according to dynamics
    """
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    
    statedot = np.zeros(6)
    statedot[0] = vx
    statedot[1] = vy
    statedot[2] = state[5]
    statedot[3] = x + 2*vy - (1-mu)*(x+mu)/(r1**3) - mu*(x-1+mu)/(r2**3)
    statedot[4] = y - 2*vx -(1-mu)*y/(r1**3) - mu*y/(r2**3)
    statedot[5] = -(1-mu)*z/(r1**3) - mu*z/(r2**3)
    
    return statedot