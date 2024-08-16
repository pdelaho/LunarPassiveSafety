import cvxpy as cp
import numpy as np

from safe_set import *

def scvx_ocp(prob):
    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time    

    if n_time > len(prob.time_hrz):
        print("Simulation time larger than problem's time horizon")
    
    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time-1, nu)) 
    l  = cp.Variable((n_time-1, nx)) # not the right shape for l
    
    # dynamics and boundary conditions
    con = []
    cost = 0 
    con += [s[0] == s_0]
    con += [s[i+1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time-1)]
    con += [s[-1] == s_f]
    
    # Unsafe ellipsoids constraint
    if prob.con_list["BRS"]:
        # prob.inv_PP = np.empty((n_time, prob.N_BRS, nx, nx))
        for i in range(n_time):
            ellipsoids = passive_safe_ellipsoid_scvx(prob, i)
            # prob.inv_PP[i] = ellipsoids
            closest, _ = extract_closest_ellipsoid(s[i], ellipsoids, 1)
            a = convexify_safety_constraint(s[i], closest, 1)
            con += [1 - np.dot(a,s[i]) <= 0]
        
    # Add the trust region constraints
    if prob.con_list["trust_region"]:
        z = s.flatten('F')
        z_bar = prob.s_ref.flatten('F')
        con += [np.linalg.norm(z_bar-z, ord='inf') <= prob.rk] # check that this line does ||z_bar - z||_inf <= r
    
    # Computing the cost L = f0 + P
    f0 = cp.sum(cp.norm(a, 2, axis=1))
    cost += f0
    P = prob.pen_λ.T @ l.flatten('F') + (prob.pen_w/2) * cp.norm(l)**2 # zeta is 0 so just this part of P is non-zero
    cost += P
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL) # cp.MOSEK
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    f0_opt  = f0.value
    P_opt = P.value
    status = p.status
    value  = p.value
    
    sol = {"mu": s_opt, "v": a_opt, "l": l_opt,"status": status, "f0": f0_opt, "P": P_opt, "value": value}
    
    return sol
    