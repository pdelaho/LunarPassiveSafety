import numpy as np
import heapq
import cvxpy as cp

# Do we want to use a given time period for passive safety or a number of time steps?
# Not a big change, just need to be consistent

def passive_safe_ellipsoid(prob, N, inv_Pf, final_time_step):
    """
        N-step Passive safety ellipoid generation. 
        return the coefficient matrix K for the passive safety ellisoid.
        at j-th step, the passive safety ellipsoid is given by:
            x.T * PP * x <= 1,
            K = Phi(kf, kf-j)^T * Pf * Phi(kf, kf-j)
        args:
            N: N-step backward reachable set (RS)
            inf_Pf: KOZ ellipsoid shape matrix (P^-1) at the terminal state (nx x nx)
            prob: RPOD OCP class
        return:
            PP: coefficient matrix for the passive safety ellipsoid ("P^-1" for the ellipsoid)
                inv_PP[j,:,:] = shape of j-step backward reachable set (RS)
    """
    nx = prob.nx
    inv_PP = np.zeros((N, nx, nx))
    # if N<len(prob.time_hrz):
    # N = prob.N_BRS 
    for j in range(N):
        Phi = prob.stm[final_time_step-j-1,:,:]

        if j > 0:
            # inv_PP[j,:,:] = Phi.T @ inv_Pf @ Phi
            inv_PP[j,:,:] = Phi.T @ inv_PP[j-1,:,:] @ Phi
        if j == 0:
            inv_PP[j,:,:] = Phi.T @ inv_Pf @ Phi
    # if N>len(prob.time_hrz):
    #     final_time_step = len(prob.time_hrz)-1
    #     for j in range(len(prob.time_hrz)):
    #         Phi = prob.stm[final_time_step-j-1,:,:]

    #         if j > 0:
    #             # inv_PP[j,:,:] = Phi.T @ inv_Pf @ Phi
    #             inv_PP[j,:,:] = Phi.T @ inv_PP[j-1,:,:] @ Phi
    #         if j == 0:
    #             inv_PP[j,:,:] = Phi.T @ inv_Pf @ Phi

    return inv_PP

def passive_safe_ellipsoid_scvx(prob, state_index):
    """
        N-step Passive safety ellipoid generation. 
        return the coefficient matrix K for the passive safety ellisoid.
        at j-th step, the passive safety ellipsoid is given by:
            x.T * PP * x <= 1,
            K = Phi(kf, kf-j)^T * Pf * Phi(kf, kf-j)
        args:
            prob: SCvx OCP class
            state_index: time step/index of the state we're looking at
        return:
            PP: coefficient matrix for the passive safety ellipsoid ("P^-1" for the ellipsoid)
                inv_PP[j,:,:] = shape of j-step backward reachable set (RS)
    """
    nx = prob.nx
    N = prob.N_BRS
    inv_PP = np.zeros((N, nx, nx))
    
    for j in range(N):
        Phi = prob.stm[state_index + N - j - 1,:,:] # should there be a -1 in here too? -> think about it

        if j > 0:
            # inv_PP[j,:,:] = Phi.T @ inv_Pf @ Phi
            inv_PP[j,:,:] = Phi.T @ inv_PP[j-1,:,:] @ Phi
        if j == 0:
            inv_PP[j,:,:] = Phi.T @ prob.inv_Pf @ Phi

    return inv_PP

def extract_closest_ellipsoid(x_ref, inv_PP_unsafe, l):
    """
    find the l closest unsafe ellipsoids from the reference trajectory 
    args: 
        x_ref: point state (nx x 1)
        inv_PP_unsafe: array of ellipsoid shape (P^-1) 
        l: number of the closest ellipsoids chosen before the convexification 
    returns:
        PP_close: closest ellipsoids' shape matrices. 
    """
    rho_sq = [x_ref.T @ inv_PP_unsafe[k,:,:] @ x_ref for k in range(inv_PP_unsafe.shape[0])]
    
    if inv_PP_unsafe.shape[0] < l:
        print("(warning) the number of the ellipoids is less than l. Check the input...")
        smallest_ele = rho_sq
    else:
        # choose the ellipsoids that are the closest to the current state (conservative)
        # smallest_ele = heapq.nsmallest(l, rho_sq)
        
        # Removing the condition that the state should be outside (ie x>1), see if I need it back
        # smallest_ele = heapq.nsmallest(l, [x for x in rho_sq]) # if x>1.0])
        # smallest_ele = cp.min([x for x in rho_sq]) # if x>1.0])
        if l == 1:
            # print(rho_sq)
            smallest_ele = min([x for x in rho_sq]) # if x>1.0])
            ind = rho_sq.index(smallest_ele)
            return [inv_PP_unsafe[ind,:,:] for i in range(1)], ind

    closest_ellipsoids = [inv_PP_unsafe[i,:,:] for i, elem in enumerate(rho_sq) if elem in smallest_ele] # and elem > 1.0)]
    indices = [i for i, elem in enumerate(rho_sq) if elem in smallest_ele] # and elem > 1.0)]
    return closest_ellipsoids, indices

def extract_closest_ellipsoid_scvx(x_ref, inv_PP_unsafe, l):
    """
    find the l closest unsafe ellipsoids from the reference trajectory 
    args: 
        x_ref: point state (nx x 1)
        inv_PP_unsafe: array of ellipsoid shape (P^-1) 
        l: number of the closest ellipsoids chosen before the convexification 
    returns:
        PP_close: closest ellipsoids' shape matrices. 
    """
    rho_sq = [x_ref.T @ inv_PP_unsafe[k,:,:] @ x_ref for k in range(inv_PP_unsafe.shape[0])]
    
    if inv_PP_unsafe.shape[0] < l:
        print("(warning) the number of the ellipoids is less than l. Check the input...")
        smallest_ele = rho_sq
    else:
        # choose the ellipsoids that are the closest to the current state (conservative)
        # smallest_ele = heapq.nsmallest(l, rho_sq)
        
        # Removing the condition that the state should be outside (ie x>1), see if I need it back
        smallest_ele = heapq.nsmallest(l, [x for x in rho_sq]) # if x>1.0])
        # smallest_ele = cp.min([x for x in rho_sq]) # if x>1.0])
        # if l == 1:
        #     # print(rho_sq)
        #     smallest_ele = min([x for x in rho_sq]) # if x>1.0])
        #     # print(smallest_ele)
        #     # ind = rho_sq.index(smallest_ele)
        #     ind = [i for i, elem in enumerate(rho_sq) if elem in smallest_ele] # and elem > 1.0)]
        #     # print('hello', ind)
        #     return [inv_PP_unsafe[ind,:,:] for i in range(1)], ind

    closest_ellipsoids = [inv_PP_unsafe[i,:,:] for i, elem in enumerate(rho_sq) if elem in smallest_ele] # and elem > 1.0)]
    indices = [i for i, elem in enumerate(rho_sq) if elem in smallest_ele] # and elem > 1.0)]
    return closest_ellipsoids, indices

def convexify_safety_constraint(x_ref, inv_PP_close, l):
    """
        Convexify the ellipsoid constraint into linear ones. 
        if l == 1: 
            only closest elliposid will be chosen. No need of second-order cone programming. 
        if l > 1: 
            requires SOCP to solve for h. 
        args: 
            x_ref: reference state (nx x 1)  *not the trajectory, but just one state
            PP_unsafe: shape matrices of the unsafe ellipsoids (a single matrix if l == 1)
            l: 
        returns: 
            h: closests unsafe elliposids => the convexified constraint is -h_jk * x_jk <= -1  (size: nx x 1)
    """
    if l == 1:       
        # x_ref = x_ref.reshape((len(x_ref),1))
        # print(inv_PP_close[0].shape, (x_ref.T @ inv_PP_close[0]).shape)
        ybar = x_ref / np.sqrt(x_ref.T @ inv_PP_close[0] @ x_ref)
        h_jk = 2 * inv_PP_close[0] @ ybar 
        
        h_jk = (x_ref.T @ inv_PP_close[0]).T / np.sqrt(x_ref.T @ inv_PP_close[0] @ x_ref)
        
    else:  
        # TODO: socp!
        raise("Not developed yet") 

    return h_jk