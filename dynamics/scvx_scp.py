import numpy as np
import scipy as sp
import cvxpy as cp
import os
import sys
from scipy.integrate import odeint

from safe_set import *
from scvx import *
from useful_small_functions import *
from dynamics_translation import *

""" Recreating the sequential convex programming algorithm SCvx* from Oguri's article"""

def compute_f0(sol,LU,TU):
    """Takes in a dictionary containing the current solution, and the length and time parameters
    in the 3-body problem, returns the value of the objective/cost function.
    Here the cost function is defined as the sum of the norm of the control actions at each timestep.

    Args:
        sol (dictionary): contains the parameters of the current solution to the problem
        LU (float): length parameter in 3-body problem
        TU (float): time parameter in 3-body problem

    Returns:
        float: value of the objective/cost function
    """
    
    # Check the dimension of a but pretty it should be dim 3 in my case
    a = sol["v"]   
    
    # Artificially multiplied by 1e10 because otherwise value is too small to get a well working SCP
    
    # TO DO: CHECK THAT THE SAME COMPUTATION OF THE COST FUNCTION IS DONE EVERYWHERE
    f0 = np.sum(np.linalg.norm(a, 2, axis=1))*1e10 # / 1e3 + cp.sum(cp.norm(a[:,3:], 2, axis=0)) 
    # f0 = np.sum(np.linalg.norm(a[:,:3]*LU/TU**2, 2, axis=0)) # / 1e3 + cp.sum(cp.norm(a[:,3:], 2, axis=0))          
    return f0


def compute_g(sol, prob):
    """To define, it is problem dependent
    Computes the equality constraints, including the dynamics (verify that fact) of the given OCP"""
    
    """Takes in the dictionary containing the current solution and the SCVX_OCP related problem, 
    returns vectors for the affine and non-convex equality constraints.

    Returns:
        n_time-1x6 vector: non-convex equality constraints linked to dynamics
        2x6 vector: affine equality constraints to enforce initial and final states
    """
    
    x = sol["mu"]
    a = sol["v"]
    g_affine = np.empty((2,x.shape[1]))
    # print(x[0]-prob.μ0)
    g_affine[0] = x[0] - prob.μ0 # initial condition
    g_affine[1] = x[-1] - prob.μf # final condition
    # Ask Yuji if the dynamics are in the affine (using linearized) or the non-affine part of the constraints
    # Need to use the non-linear dynamics, otherwise g and h=0 and P=0 so no interest in computing P=0
    # Integrate the non-linear dynamics in the moon frame to get the constraint
    g = np.empty((x.shape[0] - 1, x.shape[1])) # SHAPE NOT RIGHT
    for i in range(x.shape[0]-1):
        x_prev = lvlh_to_synodic(x[i], prob.target_traj[i], prob.mu)
        t = [0, prob.time_hrz[i+1] - prob.time_hrz[i]]
        # integrate the non-linear dynamics ADDING CONTROL, CHECK THAT IT'S DONE CORRECTLY
        x_new = odeint(dynamics_synodic_control, x_prev, t, args=(prob.mu,a[i],))
        x_new_lvlh = synodic_to_lvlh(x_new[-1], prob.target_traj[i+1], prob.mu)
        g[i] = x[i+1] - x_new_lvlh
        
    return g.flatten('F'), g_affine


# Remake some of the safe_set functions for them to be adapted to this problem
def compute_h(sol, prob): # should include what's not sol or prob in the problem class def
    """To define, it is problem dependent
    Computes the inequality constraints, includes the unsafe ellipsoids in my case"""
    
    """Takes in the dictionary containing the current solution and the SCVX_OCP related problem, 
    returns vectors for the affine and non-convex inequality constraints.

    Returns:
        1x1 vector: non-convex inequalities
        n_time-2x1 vector: convex inequlities due to passive safety (using BRS)
    """
    x = sol["mu"]
    h_cvx = np.empty((prob.n_time-2))
    # need to compute the vector perp. to the hyperspace tangent to the closest unsafe ellipsoid for each state between initial and final
    # why not at the initial state as well?
    for i in range(len(x[1:-1])):
        state = x[i+1]
        inv_PP = passive_safe_ellipsoid_scvx(prob, i)
        closest_ellipsoid, _ = extract_closest_ellipsoid_scvx(state, inv_PP, 1)
        a = convexify_safety_constraint(state, closest_ellipsoid, 1)
        h_cvx[i] = 1 - np.dot(a.reshape(6), state.reshape(6))
    h = np.zeros(1)
    
    return h, h_cvx


def get_slack(sol):    
    """Takes in a dictionar containing the current solution, returns the slack variables, differenciating the one
    used in the equality constraints and the one used in the inequality constraints.

    Args:
        sol (dictionary): contains the parameters of the current solution to the problem

    TO DO: LOOK FOR THE SIZE OF THE FOLLOWING VECTORS
    Returns:
        vector: slack variable used in the equality constraints linked to the dynamics
        vector: slack variable used in the inequality constraints linked to the passive safety
    """
    
    l = sol["l"]     
    
    # equality slack 
    ξ = l.flatten('F')
    
    # inequality slack 
    ζ = np.zeros((1))
    
    return ξ, ζ


def compute_P(g, h, w, λ, μ):
    """Takes in the constraints (both equalities and inequalities), and different weights associated with them,
    returns the value of the penalty function.

    Args:
        g (_type_): _description_
        h (_type_): _description_
        w (_type_): _description_

    Returns:
        _type_: _description_
    """
    h_pos = [i if i>=0 else 0 for i in h]
    
    return np.dot(λ, g) + (w/2)*np.linalg.norm(g)**2 + np.dot(μ, h) + (w/2)*np.linalg.norm(h_pos)**2


def update_weights(sol, prob):
    # check this function -> NOT RIGHT -> Fixed
    g_all, g_aff = compute_g(sol, prob)
    h_all, h_cvx = compute_h(sol, prob)
    # print(prob.pen_w)
    prob.pen_λ += prob.pen_w*g_all
    μ_temp = prob.pen_μ + prob.pen_w*h_all
    if μ_temp < 0:
        μ_temp = 0
    prob.pen_μ = μ_temp
    prob.pen_w *= prob.β
    
    return prob


def update_delta(δ, γ, ΔJ):
    if δ < 1e10: # if \delta > 1e10:
        δ *= γ
    else:
        δ = abs(ΔJ)
        
    return δ


def update_trust_region(r, α, ρ, ρk, r_minmax):
    α1, α2 = α
    _, ρ1, ρ2 = ρ   # ρ0 < ρ1 < ρ2
    r_min, r_max = r_minmax
    
    if ρk < ρ1:
        return np.max([r/α1, r_min])
    elif ρk < ρ2:
        return r
    else:
        return np.min([r*α2, r_max])


def compute_dJdL(sol, sol_prev, prob, iter):
    w, λ, μ = prob.pen_w, prob.pen_λ, prob.pen_μ
    ξ, ζ = get_slack(sol) #
    g_prev, _ = compute_g(sol_prev, prob)
    h_prev, _ = compute_h(sol_prev, prob)
    g, _ = compute_g(sol, prob)
    h, _ = compute_h(sol, prob)
    # h_p = np.array([ele if ele >= 0 else 0 for ele in h])
    J0 = compute_f0(sol_prev, prob.LU,prob.TU) + compute_P(g_prev, h_prev, w, λ, μ)
    J1 = compute_f0(sol,prob.LU,prob.TU) + compute_P(g, h, w, λ, μ)
    L = compute_f0(sol,prob.LU,prob.TU) + compute_P(ξ, ζ, w, λ, μ)
    # print(compute_f0(sol_prev))
    ΔJ = J0 - J1
    ΔL = J0 - L
    # print(ΔJ,ΔL)
    # print(g.shape, h_p.shape)
    # χ = np.linalg.norm(np.stack((g, h_p)))
    h_p = np.array([ele if ele >= 0 else 0 for ele in h])
    χ = np.linalg.norm(np.concatenate((g,h_p)))
    print(ΔL)
    if ΔL <= 1e-9 and iter!=0:
        # raise ValueError("ΔL must be positive! ")
        print("WARNING: ΔL must be positive! ")
        
    return ΔJ, ΔL, χ


# Check that every property of the problem class has been defined in problem_class.py
# Change the functions because N is now part of the problem class definition -> I think that's done

def scvx_star(prob, sol_0, μref, fname, max_iter=100):
    # Extract the initial parameters 
    prob.rk, prob.pen_w = prob.r0, prob.w0
    
    # Make a initial solution dictionary    
    sol_prev = sol_0
    
    # Initialization
    k = 0
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    g0, _ = compute_g(sol_prev, prob) 
    h0, _ = compute_h(sol_prev, prob)
    prob.pen_λ, prob.pen_μ = np.zeros(np.shape(g0)), np.zeros(np.shape(h0))

    # Initialization of the reference trajectory
    prob.s_ref = μref 
    # n_time = np.shape(μref)[0] # I don't think that line is necessary, already built in problem definition
    
    log = {"ΔJ":[], "χ":[], "f0":[], "P":[],
           "g":[], "h":[], 
           "s":[], "a":[], "l":[], "objective decrease":[]}
    
    prob.load_traj_data(fname)
    prob.linearize_trans()
    prob.get_unsafe_ellipsoids()
    # print(prob.hyperplanes.shape)
    while (abs(ΔJ) > prob.εopt or χ > prob.εfeas) and k < prob.iter_max:
        # linearization of the dynamics and constraints -> get the perp vector for each half plane
        for i in range(prob.n_time):
            closest, _ = extract_closest_ellipsoid_scvx(prob.s_ref[i], prob.inv_PP[i], 1)
            a = convexify_safety_constraint(prob.s_ref[i], closest, 1)
            # print(a.shape)
            prob.hyperplanes[i] = a
        
        # print(prob.hyperplanes.shape)
        # solve the convexified problem, get the suboptimal solution and slack variables
        sol = scvx_ocp(prob)

        # compute the errors
        ΔJ, ΔL, χ= compute_dJdL(sol, sol_prev, prob, k)
        
        if ΔL < 1e-4: # or prob.epsilonfeas
            ρk = 1
        else:
            ρk = ΔJ / ΔL
        
        print(f"iter: {k}, ",  ", f0:", np.round(sol["f0"],8)) # 4 instead of 8 before
        
        g, _ = compute_g(sol, prob)
        g = g.reshape((prob.n_time-1, prob.nx), order='F')
        log["ΔJ"].append(abs(ΔJ))
        log["χ"].append(χ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        log["s"].append(sol["mu"])
        log["a"].append(sol["v"])
        log["l"].append(sol["l"]) 
        log["g"].append(g)
        log["objective decrease"].append(ρk)
        
        
        if ρk >= prob.ρ[0]: # >
            # solution update
            sol_prev = sol
            prob.s_ref = sol["mu"]  # update the reference state
            print("Reference is updated")
            
            if abs(ΔJ) < δ:
                print("Weight update!")
                prob = update_weights(sol, prob)
                δ = update_delta(δ, prob.γ, ΔJ)
        
        prob.rk = update_trust_region(prob.rk, prob.α, prob.ρ, ρk, prob.r_minmax)
        
        k += 1
        
        if k > max_iter:
            print("SCvx* did not converge... terminating...")
            break
        
    return sol, log
        
        