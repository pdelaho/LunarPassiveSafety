import cvxpy as cp
import numpy as np

from safe_set import *

# ACTUAL WORK IN PROGRESS

def scvx_ocp(prob):
    """Takes in a problem statement and returns a possible optimized trajectory using Sequential 
    Convex Programming (SCP) as described in Oguri's paper (Successive Convexification with Feasibility 
    Guarantee via Augmented Lagrangian for Non-Convex Optimal Control Problems, 2024).
    We want to solve a problem where a chaser's spacecraft approaches a target spacecraft
    while ensuring passive safety using Backward Reachable Sets.

    Args:
        prob (SCVX_OCP): problem statement

    Returns:
        dictionary: regroups the follownig parameters of the optimized path: the state vectors "mu",
                    the controls "v", the slack variable for dynamics "l", the status of the solution
                    "status", the value of the objective function "f0", how well constraints are enforced
                    "P", and the value of the problem (WTF IS THAT??) "value" 
    """
    
    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time
    # μref = prob.s_ref

    if n_time > len(prob.time_hrz):
        print("Simulation time larger than problem's time horizon")
    
    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time-1, nu)) 
    l  = cp.Variable((n_time-1, nx))
    
    # dynamics and boundary conditions
    con = []
    cost = 0 
    con += [s[0] == s_0]
    con += [s[i+1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time-1)]
    con += [s[-1] == s_f]
    
    # Unsafe ellipsoids constraints
    if prob.con_list["BRS"]:
        # the computation of the unsafe ellipsoids can be done before solving the problem
        # since they only depend on the position of the target spacecraft, which is known
        # in advance, same with the linearized dynamics
        for i in range(n_time):
            con += [1 - prob.hyperplanes[i].T @ s[i].reshape((nx,1)) <= 0]
        
    # Add the trust region constraints
    if prob.con_list["trust_region"]:
        z = s.flatten('F')
        z_bar = prob.s_ref.flatten('F')
        # con += [np.linalg.norm(z_bar-z, ord='inf') <= prob.rk] # check that this line does ||z_bar - z||_inf <= r
        con += [cp.norm(z_bar-z, ord='inf') <= prob.rk] # check that this line does ||z_bar - z||_inf <= r
        # What if we apply trust region to the control as well?

    
    # Computing the cost L = f0 + P
    f0 = cp.sum(cp.norm(a, 2, axis=1))*1e10
    # f0 = cp.sum(cp.norm(a, 2, axis=0))
    cost += f0
    P = prob.pen_λ.T @ l.flatten('F') + (prob.pen_w/2) * cp.norm(l)**2 # zeta is 0 so just this part of P is non-zero
    cost += P
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL) # cp.MOSEK, CLARABEL, SCS
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    f0_opt  = f0.value
    P_opt = P.value
    status = p.status
    value  = p.value
    
    sol = {"mu": s_opt, "v": a_opt, "l": l_opt,"status": status, "f0": f0_opt, "P": P_opt, "value": value}
    
    return sol