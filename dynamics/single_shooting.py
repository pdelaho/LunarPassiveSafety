import numpy as np
import scipy.integrate as integrate
from cr3bp_barycenter import *

def single_shooting(initial_state,residual,jacobian):
    new_initial_state = initial_state.reshape((3,1)) - np.linalg.pinv(jacobian)@(residual.reshape((3,1)))
    return new_initial_state

def optimization(initial_conditions,period,max_iter=1000,tol=1e-5,step=3000):
    adjusted_conditions = initial_conditions
    tf = period/2
    t_simulation = np.linspace(0,tf,step)
    
    for i in range(max_iter):
        y_temp = integrate.odeint(halo_propagator_with_STM,adjusted_conditions,t_simulation,rtol=1e-12, atol=1e-12)
        f = np.matrix([y_temp[-1,1], y_temp[-1,3], y_temp[-1,5]])
        
        if np.linalg.norm(f)<tol:
            adjusted_conditions[0] = y_temp[0,0]
            adjusted_conditions[4] = y_temp[0,4]
            break
        else:
            # use the ode function to compute the derivatives easily
            state_end = halo_propagator(y_temp[-1,:],t_simulation[-1])
            
            df = np.matrix([[y_temp[-1,12], y_temp[-1,16], state_end[1]],
                  [y_temp[-1,24], y_temp[-1,28], state_end[3]],
                  [y_temp[-1,36], y_temp[-1,40], state_end[5]]])
            
            new_x = single_shooting(np.matrix([adjusted_conditions[0],adjusted_conditions[4],tf]),f,df)
            adjusted_conditions[0] = new_x[0,0]
            adjusted_conditions[4] = new_x[1,0]
            tf = new_x[2,0]
    
    return adjusted_conditions,tf