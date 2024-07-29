import cvxpy as cp

def ocp_cvx(prob):
    """
    General function of the OCP.
    Constraints are handled by con_list (dict).
    Cleanest implementation. Want to merge to this one eventually... (05/06)
    """

    nx, nu = prob.nx, prob.nu
    A, B = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    n_time = prob.n_time    

    # normalized vbariables 
    s = cp.Variable((n_time, nx))
    a = cp.Variable((n_time-1, nu)) 
    l  = cp.Variable((n_time-1, nx))  # slack variable to prevent artificial infeasibility
    
    # dimensionalized variables
    # s = s_ @ prob.P_s + prob.p_s.reshape((1,nx)) 
    # a = a_ @ prob.P_a + prob.p_a.reshape((1,nu))
    
    # dynamics 
    con = []
    cost = 0 
    con += [s[0] == s_0]
    con += [s[i+1] == A[i] @ s[i] + B[i] @ a[i] + l[i] for i in range(n_time-1)]
    
    if prob.control:
        con += [s[-1] == s_f]

    
    if prob.nu == 3:
        J = cp.sum(cp.norm(a, 2, axis=0))
    else:
        J = cp.sum(cp.norm(a[:,:3], 2, axis=0)) + cp.sum(cp.norm(a[:,3:], 2, axis=0))
    
    cost += J
    cost += cp.sum(cp.norm(l, 2, axis=0)) * 1e3   # slack variable penalty 
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.MOSEK, verbose=False)
    s_opt  = s.value
    a_opt  = a.value
    l_opt  = l.value    
    J_opt  = J.value
    status = p.status
    value  = p.value

    sol = {"mu": s_opt, "v": a_opt, "l": l_opt, "status": status, "control_cost": J_opt, "value": value}
    return sol
