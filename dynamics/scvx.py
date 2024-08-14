import cvxpy as cp
import numpy as np

from safe_set import *

def scvx_ocp(prob):
    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time    

    if n_time > len(prob.time_hrz):
        n_time = len(prob.time_hrz)
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
    
    # Unsafe ellipsoids constraint
    Pf = np.diag([prob.width_pos_koz**2, prob.length_pos_koz**2, prob.height_pos_koz**2, prob.width_vel_koz**2, prob.length_vel_koz**2, prob.height_vel_koz**2])
    inv_Pf = np.linalg.inv(Pf)
    ellipsoids = passive_safe_ellipsoid(prob, prob.N_BRS, inv_Pf, i+prob.N_BRS)
    for i in len(s):
        closest, _ = extract_closest_ellipsoid(s[i], ellipsoids, 1)
        a = convexify_safety_constraint(s[i], closest, 1)
        con += [1 - np.dot(a,s[i]) <= 0]
        
    # Add the trust region constraints
    z = s.flatten('F')
    z_bar = prob.s_ref.flatten('F')
    con += [np.linalg.norm(z_bar-z, ord='inf') <= prob.rk] # check that this line does ||z_bar - z||_inf <= r
        
    f0 = cp.sum(cp.norm(a, 2, axis=1))
    
    cost = J # need to add P and f0 to get the cost that we want to minimize
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.MOSEK)
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    J_opt  = J.value
    status = p.status
    value  = p.value
    
    sol = {"mu": s_opt, "v": a_opt, "l": l_opt,"status": status, "control_cost": J_opt, "value": value}
    return sol
    