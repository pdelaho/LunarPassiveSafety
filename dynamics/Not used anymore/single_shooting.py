import numpy as np
import scipy.integrate as integrate


from cr3bp_barycenter import halo_propagator, halo_propagator_with_STM


def single_shooting(initial_state, residual, jacobian):
    """Takes in initial conditions for an orbit, the residuals after propagating the dynamics for half a period, 
    and the jacobian, and returns the changed initial conditions 

    Args:
        initial_state (3x1 vector): initial conditions for an orbit
        residual (3x1 vector): residuals after propagation
        jacobian (3x3 matrix): jacobian of the dynamics

    Returns:
        3x1 vector: adjusted initial conditions
    """
    
    new_initial_state = initial_state.reshape((3,1)) - np.linalg.pinv(jacobian) @ (residual.reshape((3,1)))
    
    return new_initial_state


def optimization(initial_conditions, period, mu, max_iter=1000, tol=1e-5, step=3000):
    """Takes in some initial conditions for an orbit, the associated period, the mass ratio parameter, the maximum number of 
    iterations allowed, the tolerance criteria, and the number of time steps we want when propagating the orbit.
    It returns the new adjusted initial conditions for both x, y, and z, and for the period.
    
    Args:
        initial_conditions (6x1 vector): initial conditions for the orbit
        period (scalar): period of the orbit
        mu (scalar): mass ratio parameter in 3-body problem
        max_iter (int, optional): maximum number of iterations to converge. Defaults to 1000.
        tol (scalar, optional): tolerance at which we consider it converged. Defaults to 1e-5.
        step (int, optional): number of time steps when propagating the orbit. Defaults to 3000.

    Returns:
        6x1 vector: adjusted initial conditions
        scalar: adjusted period given the adjusted initial conditions
    """
    
    adjusted_conditions = initial_conditions
    tf = period / 2
    t_simulation = np.linspace(0, tf, step)
    max_iter = 1000
    
    for i in range(max_iter):
        y_temp = integrate.odeint(halo_propagator_with_STM, adjusted_conditions, t_simulation, args=(mu,), rtol=1e-12, atol=1e-12)
        f = np.matrix([y_temp[-1,1], y_temp[-1,3], y_temp[-1,5]])
        
        if np.linalg.norm(f) < tol:
            adjusted_conditions[0] = y_temp[0,0]
            adjusted_conditions[4] = y_temp[0,4]
            break
        
        else:
            # use the ode function to compute the derivatives easily
            state_end = halo_propagator(y_temp[-1,:], t_simulation[-1], args=(mu,))
            
            df = np.matrix([
                [y_temp[-1,12], y_temp[-1,16], state_end[1]],
                [y_temp[-1,24], y_temp[-1,28], state_end[3]],
                [y_temp[-1,36], y_temp[-1,40], state_end[5]]
                ])
            
            new_x = single_shooting(np.matrix([adjusted_conditions[0], adjusted_conditions[4], tf]), f, df)
            adjusted_conditions[0] = new_x[0,0]
            adjusted_conditions[4] = new_x[1,0]
            tf = new_x[2,0]
    
    return adjusted_conditions, tf