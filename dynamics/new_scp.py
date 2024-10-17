"""
SCVx* implemenmtation... (Oguri, 2023) for passive safety proximity operations
"""

import numpy as np
import cvxpy as cp

from dynamics_translation import *
from safe_set import *

def ocp_cvx_AL(prob, verbose=False):
    """
    General function of the OCP.
    Constraints are handled by con_list (dict).
    It takes in a problem from the SCVX_OCP class
    and returns a solution to the small convexified subproblem
    """

    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time
    sbar = prob.s_ref.flatten('F') # I can also flatten later, only when needed
    abar = prob.control_actions.flatten('F')
    
    if n_time > len(prob.time_hrz):
        print("Simulation time larger than problem's time horizon")
    
    zbar = np.concatenate((sbar,abar))    

    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time-1, nu)) 
    l = cp.Variable((n_time-1, nx))
    z = np.concatenate((s.flatten('F'),a.flatten('F')))
    
    # dynamics and boundary conditions
    con = []
    cost = 0 
    # linearized dynamics
    con += [s[0] == s_0]
    con += [s[i+1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time-1)]
    con += [s[-1] == s_f]
    
    if prob.con_list["BRS"]:
        for i in range(n_time):
            con += [1 - prob.hyperplanes[i].T @ s[i].reshape((nx,1)) <= 0]
    
    if prob.con_list["trust_region"]:
        zbar = np.concatenate((sbar,abar))
        z = np.concatenate((s.flatten('F'),a.flatten('F')))
        con += [cp.norm(zbar-z, ord='inf') <= prob.rk] # check that this line does ||z_bar - z||_inf <= r_k

    g = ξ   # column-wise vectorization 
    h = np.zeros((1))
    f0 = compute_f0(z)
    P  = compute_P(g, h, prob.pen_w, prob.pen_λ, prob.pen_μ)    
    cost = f0 + P 
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL, verbose=verbose)
    z_opt  = z.value 
    status = p.status
    f0_opt = f0.value
    P_opt  = P.value
    L_opt  = p.value
    ξ_opt  = ξ.value    
    ζ_opt  = h   

    sol = {"z": z_opt, "ξ": ξ_opt, "ζ": ζ_opt,  "status": status, "L": L_opt, "f0": f0_opt, "P": P_opt}
    
    return sol


def solve_cvx_AL(prob, verbose=False):  
    """
    Solving the convexified problem. 
    You may add any convexification process here (e.g., comptuation of state transition matrix in the nonlinear dynamics...).
    """

    sol = ocp_cvx_AL(prob, verbose=verbose)
    
    return sol 



# define objective functions 
        
def compute_f0(a, prob=None):
    """
    Objective function (written in CVXPY)
    """
    
    return cp.sum(cp.norm(a,2,axis=1))*1e10   # need to rescale control actions so that the cost is not too low


def compute_P(g, h, pen_w, pen_λ, pen_μ):
    """
    Compute the (convex) penalty term for the argumented Lagrangian (written in CVXPY)
    NO NEED TO CHANGE THIS FUNCTION.
    """
    
    zero = cp.Constant((np.zeros(h.shape)))
    hp = cp.maximum(zero, h)
    
    P = pen_λ.T @ g + pen_μ.T @ h + pen_w/2 * (cp.norm(g)**2 + cp.norm(hp)**2)
    
    return P


def compute_g(s, a, prob=None):
    """ 
    Returning nonconvex constraint value g and convex constraint value g_affine
    """
    g_affine = np.zeros((2,prob.nx))
    g_affine = s[0] - prob.μ0 # initial condition
    g_affine = s[-1] - prob.μf # final condition
    
    g = np.empty((s.shape[0] - 1, s.shape[1]))
    for i in range(s.shape[0]-1):
        x_prev = lvlh_to_synodic(s[i], prob.target_traj[i], prob.mu)
        t = [0, prob.time_hrz[i+1] - prob.time_hrz[i]]
        x_new = odeint(dynamics_synodic_control, x_prev, t, args=(prob.mu,a[i],))
        x_new_lvlh = synodic_to_lvlh(x_new[-1], prob.target_traj[i+1], prob.mu)
        g[i] = s[i+1] - x_new_lvlh
    
    return g, g_affine


def compute_h(s, prob=None):
    """
    Return nonconex inequalities h and convex inequalities h_cvx 
    """
    h = np.zeros((s.shape[0]))
    
    for i in range(s.shape[0]):
        state = s[i+1]
        inv_PP = passive_safe_ellipsoid_scvx(prob, i)
        closest_ellipsoid, _ = extract_closest_ellipsoid_scvx(state, inv_PP, 1)
        distance = state.T @ closest_ellipsoid @ state # the non convex inequality is just being outside the ellipsoid
        h[i] = 1 - distance

    h_cvx = np.zeros(1) # check the size
    
    return h, h_cvx