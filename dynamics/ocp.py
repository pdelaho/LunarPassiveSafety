import cvxpy as cp

# TO DO: check if we really need 2 differents ocp functions for what I use them for

def ocp_cvx(prob):
    """
    General function of the OCP.
    Constraints are handled by con_list (dict).
    Cleanest implementation. Want to merge to this one eventually... (05/06)
    """

    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time    

    if n_time > len(prob.time_hrz):
        n_time = len(prob.time_hrz)
    
    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time - 1, nu)) 
    l  = cp.Variable((n_time - 1, nx))  # slack variable to prevent artificial infeasibility
    
    # dynamics 
    con = []
    cost = 0 
    con += [s[0] == s_0]
    con += [s[i + 1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time - 1)]
    
    if prob.control:
        con += [s[-1] == s_f]
    
    if prob.nu == 3:
        J = cp.sum(cp.norm(a, 2, axis=0))
    else:
        J = cp.sum(cp.norm(a[:, :3], 2, axis=0)) * 1e10 + cp.sum(cp.norm(a[:, 3:], 2, axis=0)) * 1e10
    
    cost += J
    cost += cp.sum(cp.norm(l, 2, axis=0)) * 1e5   # slack variable penalty
    
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.MOSEK, verbose=False) # or MOSEK
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    J_opt  = J.value
    status = p.status
    value  = p.value

    sol = {"mu": s_opt, "v": a_opt, "l": l_opt, "status": status, "control_cost": J_opt, "value": value}
    
    return sol

def ocp_cvx_scvx(prob):
    """
    General function of the OCP.
    Constraints are handled by con_list (dict).
    Cleanest implementation. Want to merge to this one eventually... (05/06)
    """

    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    n_time   = prob.n_time    

    if n_time > len(prob.time_hrz):
        n_time = len(prob.time_hrz)
    
    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time - 1, nu)) 
    l  = cp.Variable((n_time - 1, nx))  # slack variable to prevent artificial infeasibility
    
    # dynamics 
    con = []
    cost = 0 
    con += [s[0] == s_0]
    con += [s[i + 1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time - 1)]
    
    if prob.control:
        con += [s[-1] == s_f]
    
    cost += cp.sum(cp.norm(l, 2, axis=0)) * 1e5   # slack variable penalty, * 1e3
    
    # to compute the cost the way it is done in the scvx optimization algorithm
    J = cp.sum(cp.norm(a, 2, axis=1)) * 1e10
    cost += J
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL, verbose=False) # or MOSEK
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    J_opt  = J.value
    status = p.status
    value  = p.value

    sol = {"s": s_opt, "a": a_opt, "l": l_opt, "status": status, "control_cost": J_opt, "value": value}
    
    return sol