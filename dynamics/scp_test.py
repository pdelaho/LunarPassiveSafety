import numpy as np
import cvxpy as cp
from scipy.integrate import odeint

from useful_small_functions import *
from dynamics_translation import *

def ocp_cvx_AL(prob, verbose=False):
    nx, nu   = prob.nx, prob.nu
    A, B     = prob.stm, prob.cim # generalized dynamics 
    s_0, s_f = prob.μ0, prob.μf
    s_ref = prob.s_ref # assuming that s is size (n_time, nx)
    a_ref = prob.a_ref # assuming that a is size (n_time-1, nu)
    z_ref = np.concatenate(s_ref.flatten("C"),a_ref.flatten("C"))
    n_time   = prob.n_time
    
    # normalized vbariables 
    # s = cp.Variable((n_time, nx)) # state/trajectory
    # a = cp.Variable((n_time-1, nu)) # controls
    # Putting both the states/trajectory and controls in the same vector because we're looking for both
    z = cp.Variable((nx*n_time + nu*(n_time-1),)) # first the states and then the controls
    ξ  = cp.Variable((n_time-1, nx))
    
    con = []
    # trajectory and dynamics constraints (convex)
    con += [z[0:nx] - s_0 == 0]
    # transpose the slices because z is supposed to be a 1D line vector and we need column vectors to do these operations with the matrices
    con += [z[nx*(i+1):nx*(i+2)] - A[i] @ z[nx*i:nx*(i+1)].T - B[i] @ z[nx*n_time + nu*i:nx*n_time + nu*(i+1)].T == ξ[i].T for i in range(n_time-1)]
    con += [z[nx*(n_time-1):nx*n_time] - s_f == 0]
    
    # nonconvex constraints due to BRS
    # added later
    
    # trust region constraint
    # z and z_ref should both be 1D and same length of course
    con += [cp.norm(z_ref - z,"inf") <= prob.rk]
    
    g = ξ
    h = np.zeros((1)) # check the dimension of h
    f0 = compute_f0(z)
    P = compute_P(g, h, prob.pen_w, prob.pen_λ, prob.pen_μ)
    cost = f0 + P
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL, verbose=verbose)
    # s_opt = s.value
    # a_opt = a.value
    z_opt = z.value
    status = p.status
    f0_opt = f0.value
    P_opt = P.value
    L_opt = p.value # WTF??
    ξ_opt = ξ.value
    ζ_opt = h
    
    sol = {"z": z_opt, "ξ": ξ_opt, "ζ": ζ_opt, "status": status, "L": L_opt, "f0": f0_opt, "P": P_opt}
    
    return sol

def solve_cvx_AL(prob):
    
    sol = ocp_cvx_AL(prob)
    
    return sol

def compute_f0(z, prob=None):
    
    a = z[prob.nx*prob.n_time:]
    
    return cp.sum(a)

def compute_P(g, h, pen_w, pen_λ, pen_μ):
    zero = cp.Constant((np.zeros(h.shape)))
    hp = cp.maximum(zero,h)
    
    P = pen_λ.T @ g + pen_μ.T @ h + pen_w/2 * (cp.norm(g)**2 + cp.norm(hp)**2)
    
    return P

def compute_g(z, prob=None):
    nx = prob.nx
    nu = prob.nu
    n_time = prob.n_time
    g_affine = np.array([z[0:nx]-prob.μ0, z[nx*(n_time-1):nx*n_time]-prob.μf])
    g =  np.zeros((n_time-1,6))
    # for every point on the trajectory (except the last one), put it
    # in the synodic frame and integrate the non-linear dynamics until
    # the next time step (-> see how to get it)
    for i in range(n_time-1):
        state_synodic = lvlh_to_synodic(z[nx*i:nx*(i+1)],prob.target_traj[i],prob.mu)
        t = [0, prob.time_hrz[i+1] - prob.time_hrz[i]]
        new_state_synodic = odeint(dynamics_synodic_control,state_synodic,t,args=(prob.mu,z[nx*n_time + nu*i:nx*n_time + nu*(i+1)]))
        new_state_lvlh = synodic_to_lvlh(new_state_synodic,prob.target_traj[i+1],prob.mu)
        g[i] = new_state_lvlh - z[nx*(i+1):nx*(i+2)]
        
    return g, g_affine

def compute_h(z, prob=None):
    # ask Yuji if I should just consider that I have convex constraints
    # since the beginning or if I consider the non convex too and then
    # h_tilde only appears when solving the convexified subproblem
    h = ...
    h_cvx = ...
    return h, h_cvx

def update_weights(z, prob):
    g, _ = compute_g(z, prob)
    h, _ = compute_h(z, prob)
    
    prob.pen_λ += prob.pen_w * g
    prob.pen_μ += prob.pen_w * h
    prob.pen_μ[prob.pen_μ < 0.0] = 0
    prob.pen_w *= prob.β
    
    return prob

def update_delta(δ, γ, ΔJ):
    return γ * δ if δ < 1e10 else abs(ΔJ)

def update_r(r, α, ρ, ρk, r_minmax):
    α1, α2 = α
    _, ρ1, ρ2 = ρ
    r_min, r_max = r_minmax
    
    if ρk < ρ1:
        return np.max([r/α1, r_min])
    elif ρk < ρ2:
        return r
    else:
        return np.min([α2*r, r_max])
    
def compute_dJdL(sol, sol_prev, prob, iter):
    w, λ, μ = prob.pen_w, prob.pen_λ, prob.pen_μ
    
    g_sol, _ = compute_g(sol["z"], prob)
    h_sol, _ = compute_h(sol["z"], prob)
    
    g_sol_prev, _ = compute_g(sol_prev["z"], prob)
    h_sol_prev, _ = compute_h(sol_prev["z"], prob)
    
    J0 = compute_f0(sol["z"], prob).value + compute_P(g_sol, h_sol, w, λ, μ).value
    J1 = compute_f0(sol_prev["z"], prob).value + compute_P(g_sol_prev, h_sol_prev, w, λ, μ).value
    L = compute_f0(sol["z"], prob).value + compute_P(sol["ξ"], sol["ζ"], w, λ, μ).value
    
    ΔJ = J1 - J0
    ΔL = J1 - L
    
    h_p = np.array([ele if ele >= 0 else 0 for ele in h_sol])
    # check the following step given my dimensions and formatting of my
    # variables
    χ = np.linalg.norm(np.hstack((g_sol, h_p)))
    
    assert np.abs(L - sol["L"]) < 1e-6, "L must be equal to the one computed in cvxpy! "
    
    print("J_new: ", np.round(J0,4), ", J_prev: ", np.round(J1,4), ", L_new: ", np.round(L,4))
    
    if ΔL < 0 and iter!=0:
        print("WARNING: ΔL must be positive!")
    
    return ΔJ, ΔL, χ

def solve_scvx(prob):
    # get initial parameters
    prob.rk, prob.pen_w = prob.r0, prob.w0
    
    sol_prev = prob.sol_0
    
    k = 0 # or 1 like in the article
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    g, _ = compute_g(sol_prev["z"], prob)
    h, _ = compute_h(sol_prev["z"], prob)
    prob.pen_λ, prob.pen_μ = np.zeros(np.shape(g)), np.zeros(np.shape(h))
    
    log = {"ϵopt":[], "ϵfeas":[], "f0":[], "P":[]}
    
    while abs(ΔJ) > prob.εopt or χ > prob.εfeas:
        
        sol = solve_cvx_AL(prob, verbose=False)
        
        if sol["status"] == "infeasible":
            print("infeasible iCS! terminating...")
            break
        
        ΔJ, ΔL, χ = compute_dJdL(sol, sol_prev, prob, k)
        
        if ΔL < prob.εfeas:
            ρk = 1
        else:
            ρk = ΔJ / ΔL
            
        print(f"iter: {k}, ", "L(=f0+P):", np.round(sol["L"], 4),  ", f0:", np.round(sol["f0"],4), ", P:", np.round(sol["P"],4), f", χ: {χ:.3f},  r: {prob.rk:.6f}, w: {prob.pen_w:.1f}, ρk: {ρk:.3f}, ΔJ: {ΔJ:.3f}, ΔL: {ΔL:.3f}, δ: {δ:.3f}")
        
        log["εfeas"].append(χ)
        log["εopt"].append(ΔJ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        
        if ρk >= prob.ρ[0]:
            
             print("reference is updated... ")
             sol_prev = sol
             prob.zref = sol["z"]
             
             if abs(ΔJ) < δ:
                 print('======= weight update! =========')
                 prob = update_weights(sol["z"],prob)
                 δ = update_delta(δ, prob.γ, ΔJ)
        
        prob.rk = update_r(prob.rk, prob.α, prob.ρ, ρk, prob.r_minmax)
        
        k +=1
        
        if k > prob.iter_max:
            print("SCVx* did not converge... terminating...")
            break
        
    return sol, log