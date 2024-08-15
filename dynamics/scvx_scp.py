import numpy as np
import scipy as sp
import cvxpy as cp
import os
import sys

from safe_set import *
from scvx import *
from useful_small_functions import *

""" Recreating the sequential convex programming algorithm SCvx* from Oguri's article"""

def compute_f0(sol):
    # Check the dimension of a but pretty it should be dim 3 in my case
    a = sol["v"]   
    f0 = cp.sum(cp.norm(a[:,:3], 2, axis=0)) # / 1e3 + cp.sum(cp.norm(a[:,3:], 2, axis=0)) 
    
    return f0


def compute_g(sol, prob):
    """To define, it is problem dependent
    Computes the equality constraints, including the dynamics (verify that fact) of the given OCP"""
    x = sol["mu"]
    g_affine = np.empty(2)
    g_affine[0] = x[0] - prob.μ0 # initial condition
    g_affine[1] = x[-1] - prob.μf # final condition
    # Ask Yuji if the dynamics are in the affine (using linearized) or the non-affine part of the constraints
    
    return g_affine


# Remake some of the safe_set functions for them to be adapted to this problem
def compute_h(sol, prob, inv_Pf, final_time_step): # should include what's not sol or prob in the problem class def
    """To define, it is problem dependent
    Computes the inequality constraints, includes the unsafe ellipsoids in my case"""
    x = sol["mu"]
    h_cvx = np.empty((prob.n_time-2)) # or prob.n_time-1?
    # need to compute the vector perp. to the hyperspace tangent to the closest unsafe ellipsoid for each state between initial and final
    for i in range(len(x[1:-1])):
        state = x[i+1]
        inv_PP = passive_safe_ellipsoid(prob, prob.N_BRS, inv_Pf, final_time_step)
        closest_ellipsoid, _ = extract_closest_ellipsoid(state, inv_PP, 1)
        a = convexify_safety_constraint(state, closest_ellipsoid, 1)
        h_cvx[i] = 1 - np.dot(a, state)
        
    return h_cvx


def get_slack(sol):    

    l     = sol["l"]    
    # χ_fov = sol["χ_fov"]
    # χ_dw  = sol["χ_dw"]  
    
    # equality slack 
    ξ = l.flatten('F')
    
    # inequality slack 
    ζ = np.zeros((1))
    
    return ξ, ζ


def compute_P(g, h, w, λ, μ):
    h_pos = [i if i>=0 else 0 for i in h]
    
    return np.dot(λ, g) + (w/2)*np.linalg.norm(g)**2 + np.dot(μ, h) + (w/2)*np.linalg.norm(h_pos)**2


def update_weights(sol, prob, inv_Pf, final_time_step):
    g_all = compute_g(sol, prob) # compute the equality constraints, maybe use compute_g depending on how it is defined
    h_all = compute_h(sol, prob, inv_Pf, final_time_step)
    prob.λ += np.dot(prob.w, g_all)
    prob.μ += np.dot(prob.w, h_all)
    if μ_temp < 0:
        μ_temp = 0
    prob.w *= prob.β
    
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
    w, λ, μ = prob.w, prob.λ, prob.μ
    ξ, ζ = get_slack(sol)
    g_prev = compute_g(sol_prev, prob)
    h_prev = 0
    g = compute_g(sol, prob)
    h = 0
    h_p = np.array([ele if ele >= 0 else 0 for ele in h])
    J0 = compute_f0(sol_prev) + compute_P(g_prev, h_prev, w, λ, μ)
    J1 = compute_f0(sol) + compute_P(g, h, w, λ, μ)
    L = compute_f0(sol) + compute_P(ξ, ζ, w, λ, μ)
    
    ΔJ = J0 - J1
    ΔL = J0 - L
    
    χ = np.linalg.norm(np.stack((g, h_p)))
    
    if ΔL < 0 and iter!=0:
        # raise ValueError("ΔL must be positive! ")
        print("WARNING: ΔL must be positive! ")
        
    return ΔJ, ΔL, χ


# Check that every property of the problem class has been defined in problem_class.py
# Change the functions because N is now part of the problem class definition -> I think that's done

def scvx_star(prob, sol_0, inv_Pf, final_time_step, μref, fname, max_iter=100):
    # Extract the initial parameters 
    prob.rk, prob.pen_w = prob.r0, prob.w0
    
    # Make a initial solution dictionary    
    sol_prev = sol_0
    
    # Initialization
    k = 0
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    g0, h0 = compute_g(sol_prev, prob), compute_h(sol_prev, prob, inv_Pf, final_time_step)
    prob.pen_λ, prob.pen_μ = np.zeros(np.shape(g0)), np.zeros(np.shape(h0))

    # Initialization of the reference trajectory
    prob.s_ref = μref   # need for the trust region constraint 
    # n_time = np.shape(μref)[0] # I don't think that line is necessary, already built in problem definition
    
    log = {"ΔJ":[], "χ":[], "f0":[], "P":[],
           "g":[], "h":[], 
           "s":[], "a":[], "l":[], "objective decrease":[]}
    
    prob.load_traj_data(fname)
    prob.linearize_trans()

    while abs(ΔJ) > prob.εopt or χ > prob.εfeas:
        # compute linearized g about the last iteration solution -> linearize dynamics + add the correct constraints in the solving function
        g_tilde = ...
        # integrated in the solving function for now, no need to linearize h
        
        # solve the convexified problem, get the suboptimal solution and slack variables
        sol = scvx_ocp(prob)
        
        # compute the errors
        ΔJ, ΔL, χ = compute_dJdL(sol, sol_prev, prob, k)
        
        if ΔL < 1e-4:
            ρk = 1
        else:
            ρk = ΔJ / ΔL
        
        g = compute_g(sol, prob).reshape((prob.n_time-1, prob.nx), order='F') # see if need to reshape
        log["ΔJ"].append(abs(ΔJ))
        log["χ"].append(χ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        log["s"].append(sol["mu"])
        log["a"].append(sol["v"])
        log["l"].append(sol["l"]) 
        log["g"].append(g)
        log["objective decrease"].append(ρk)
        
        
        if ρk > prob.ρ[0]:
            # solution update
            sol_prev = sol
            prob.s_ref = sol["mu"]  # update the reference state
            print("Reference is updated")
            
            if abs(ΔJ) < δ:
                prob = update_weights(prob)
                δ = update_delta(δ, prob.γ, ΔJ)
        
        prob.rk = update_trust_region(prob.rk, prob.α, prob.ρ, ρk, prob.r_minmax)
        
        k += 1
        
        if k > max_iter:
            print("SCVx* did not converge... terminating...")
            break
        
    return prob, log
        
        