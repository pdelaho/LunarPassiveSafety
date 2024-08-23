# Reproduction of the first example of Oguri's article about the scvx* algorithm
import numpy as np
import cvxpy as cp

class Example1:
    def __init__(self,
                 w0=0.1):
        
        # SCvx parameters, tune them to make the code run
        self.α = np.array([2, 3])
        self.β = 2
        self.γ = 0.9
        self.ρ = np.array([-0.01, 0.25, 0.7])
        self.r_minmax = np.array([1e-10, 10])
        self.εopt = 1e-4 # they use 1e-5 in the article
        self.εfeas = 1e-4
        self.pen_λ = None
        self.pen_μ = None
        self.pen_w = None
        self.rk = None
        self.r0 = 2
        self.w0 = w0
        
        # Reference solution
        self.z_ref = None

def compute_f0_ex1(sol):
    z = sol["z"]
    return z[0] + z[1]

def compute_g_ex1(sol):
    z = sol["z"]
    return np.array([z[1] - z[0]**4 - 2*z[0]**3 + 1.2*z[0]**2 + 2*z[0]]), 0

def compute_h_ex1(sol):
    z = sol["z"]
    return 0, np.array([z[0]-2, -z[0]-2, z[1]-2, -z[1]-2, -z[1] - (4/3)*z[0] - 2/3])

def compute_grad_g_ex1(sol):
    z = sol["z"]
    z = z.reshape(2)
    # print(z.reshape(2))
    return np.asarray([-4*z[0]**3 - 6*z[0]**2 + 2.4*z[0] + 2, 1])

def get_slack(sol):    
    l = sol["l"]
    # equality slack 
    ξ = l.flatten('F')
    # inequality slack 
    ζ = np.zeros((1))
    
    return ξ, ζ

def compute_P(g, h, w, λ, μ):
    if h != 0:
        h_pos = [i if i>=0 else 0 for i in h]
    else:
        h_pos = 0
    
    return np.dot(λ, g) + (w/2)*np.linalg.norm(g)**2 + np.dot(μ, h) + (w/2)*np.linalg.norm(h_pos)**2

def update_weights(sol, prob):
    g_all, _ = compute_g_ex1(sol)
    h_all, _ = compute_h_ex1(sol)
    prob.pen_λ += prob.pen_w*g_all.reshape(1)
    μ_temp = prob.pen_μ + prob.pen_w*h_all
    if μ_temp < 0:
        μ_temp = 0
    prob.pen_μ = μ_temp
    prob.pen_w *= prob.β
    
    return prob

def update_delta(δ, γ, ΔJ):
    if δ > 1e10:
        δ *= γ
    else:
        δ = abs(ΔJ)
        
    return δ

def update_trust_region(r, α, ρ, ρk, r_minmax):
    α1, α2 = α
    _, ρ1, ρ2 = ρ   # ρ0 < ρ1 < ρ2
    r_min, r_max = r_minmax
    
    if ρk < ρ1:
        return max([r/α1, r_min])
    elif ρk < ρ2:
        return r
    else:
        return min([r*α2, r_max])
    
def compute_dJdL(sol, sol_prev, prob, iter):
    w, λ, μ = prob.pen_w, prob.pen_λ, prob.pen_μ
    ξ, ζ = get_slack(sol)
    g_prev, _ = compute_g_ex1(sol_prev)
    h_prev, _ = compute_h_ex1(sol_prev)
    g, _ = compute_g_ex1(sol)
    h, _ = compute_h_ex1(sol)
    if h != 0:
        h_p = np.array([ele if ele >= 0 else 0 for ele in h])
    else:
        h_p = 0
    J0 = compute_f0_ex1(sol_prev) + compute_P(g_prev, h_prev, w, λ, μ)
    J1 = compute_f0_ex1(sol) + compute_P(g, h, w, λ, μ)
    L = compute_f0_ex1(sol) + compute_P(ξ, ζ, w, λ, μ)

    ΔJ = J0 - J1
    ΔL = J0 - L
    # print(g.reshape(1), np.asarray(h_p).reshape(1))
    χ = np.linalg.norm(np.concatenate((g.reshape(1),np.asarray(h_p).reshape(1))))
    print(ΔL)
    if ΔL <= -1e-9 and iter!=0:
        # raise ValueError("ΔL must be positive! ")
        print("WARNING: ΔL must be positive! ")
        
    return ΔJ, ΔL, χ

def scvx_ocp_ex1(prob, g, grad_g, sol):
    z = cp.Variable((2,1))
    l = cp.Variable((1,1))
    con = []
    
    # equalities constraints
    con += [g + grad_g.reshape((1,2)) @ (z.reshape((2,1)) - sol["z"].reshape((2,1))) == l]
    
    # inequalities constraints
    con += [z[0] - 2 <= 0]
    con += [-z[0] - 2 <= 0]
    con += [z[1] - 2 <= 0]
    con += [-z[1] - 2 <= 0]
    con += [-z[1] - (4/3)*z[0] - 2/3 <= 0]
        
    cost = 0
    f0 = z[0] + z[1]
    cost += f0
    P = prob.pen_λ.T @ l.flatten('F') + (prob.pen_w/2) * cp.norm(l)**2 # zeta is 0 so just this part of P is non-zero
    cost += P
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.MOSEK) # cp.MOSEK, CLARABEL, SCS
    z_opt  = z.value
    l_opt  = l.value    
    f0_opt  = f0.value
    P_opt = P.value
    status = p.status
    value  = p.value
    
    sol = {"z": z_opt, "l": l_opt, "status": status, "f0": f0_opt, "P": P_opt, "value": value}
    
    return sol

def scvx_star(prob, sol_0, zref, max_iter=100):
    # Extract the initial parameters 
    prob.rk, prob.pen_w = prob.r0, prob.w0
    
    # Make a initial solution dictionary    
    sol_prev = sol_0
    
    # Initialization
    k = 0
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    g0, _ = compute_g_ex1(sol_prev) 
    h0, _ = compute_h_ex1(sol_prev)
    prob.pen_λ, prob.pen_μ = np.zeros(np.shape(g0)), np.zeros(np.shape(h0))

    # Initialization of the reference trajectory
    prob.z_ref = zref 
    # n_time = np.shape(μref)[0] # I don't think that line is necessary, already built in problem definition
    
    log = {"ΔJ":[], "χ":[], "f0":[], "P":[],
           "g":[], "h":[], 
           "z":[], "l":[], "objective decrease":[]}
    
    while abs(ΔJ) > prob.εopt or χ > prob.εfeas:
        # linearization of the constraints
        grad_g = compute_grad_g_ex1(sol_prev)
        g, _ = compute_g_ex1(sol_prev)
        # _, h_cvx = compute_h_ex1(sol_prev)
        
        # solve the convexified problem, get the suboptimal solution and slack variables
        sol = scvx_ocp_ex1(prob, g, grad_g, sol_prev)
        
        # compute the errors
        ΔJ, ΔL, χ = compute_dJdL(sol, sol_prev, prob, k)
        # print(ΔL<1e-4)
        if ΔL < 1e-4:
            ρk = 1
        else:
            ρk = ΔJ / ΔL
            print(ρk)
        
        # g, _ = compute_g_ex1(sol)
        # g = g.reshape((prob.n_time-1, prob.nx), order='F') # g should be dim 1 here
        log["ΔJ"].append(abs(ΔJ))
        log["χ"].append(χ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        log["z"].append(sol["z"])
        # log["a"].append(sol["v"])
        log["l"].append(sol["l"]) 
        log["g"].append(g)
        log["objective decrease"].append(ρk)
        
        
        if ρk > prob.ρ[0]:
            # solution update
            sol_prev = sol
            prob.z_ref = sol["z"]  # update the reference state
            print("Reference is updated")
            
            if abs(ΔJ) < δ:
                prob = update_weights(sol, prob)
                δ = update_delta(δ, prob.γ, ΔJ)
        
        prob.rk = update_trust_region(prob.rk, prob.α, prob.ρ, ρk, prob.r_minmax)
        
        k += 1
        
        if k > max_iter:
            print("SCvx* did not converge... terminating...")
            break
        
    return prob, log, k