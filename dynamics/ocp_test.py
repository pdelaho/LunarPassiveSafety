import numpy as np 
import sys
import os 
import matplotlib.pyplot as plt

from useful_small_functions import *
from dynamics_linearized import *
from ocp import * 


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

fname = root_folder + "/dynamics/data/9_2_S_Halo.json"

t, traj, mu, LU, TU = load_traj_data(fname)
print("traj loaded...")
print(f"mu: {mu} km^3/s^2")   
print(f"LU: {LU} km")
print(f"TU: {TU} s")  

mats = linearize_translation(mu, traj, t, True)

class OCP:
    def __init__(self, mats):
        self.stm = mats["stm"]
        self.cim = mats["cim"]
        self.psi = mats["psi"]  # LVLH -> Synodic rotation matrix
        self.μ0 = None
        self.μf = None
        self.n_time = None
        self.nu = 3
        self.nx = 6 


prob = OCP(mats)

# boundary condition (in km, and non-dimensionalized by LU)
# LVLH [i,j,k] = [T, -N, -R]
prob.μ0 = np.array([10,0,0,0,0,0])  / LU 
prob.μf = np.array([0,0,0,0,0,0])  / LU
prob.n_time = 200 #len(t)

sol = ocp_cvx(prob) 
mu  = sol["mu"] 
J   = sol["control_cost"]    
print(f"J (control cost): {J}")

# plot the absolute trajectory (NRHO)
fig = plt.figure(figsize=(10,8)) 
ax  = fig.add_subplot(111, projection='3d') 
ax.plot3D(traj[:,0], traj[:,1], traj[:,2], 'k') 
ax.scatter3D(traj[0,0], traj[0,1], traj[0,2], c='r', s=100, label='start')  # initial state is apoapsis
ax.scatter3D(1,0,0, c='orange', s=100, label='Moon')
ax.axis('equal')
ax.set_xlabel('x, LU')
ax.set_ylabel('y, LU') 
ax.set_zlabel('z, LU')
plt.legend() 


# plot the relative trajecotry (scaled in km) in the RTN frame # LVLH [i,j,k] = [T, -N, -R]
fig = plt.figure(figsize=(10,8)) 
ax  = fig.add_subplot(111, projection='3d') 
ax.plot3D(mu[:,0]*LU, -mu[:,1]*LU, -mu[:,2]*LU, 'k') 
ax.scatter3D(mu[0,0]*LU,  -mu[0,1]*LU,  -mu[0,2]*LU,  c='r', s=100, label='start')
ax.scatter3D(mu[-1,0]*LU, -mu[-1,1]*LU, -mu[-1,2]*LU, c='b', s=100, label='end')
ax.set_xlabel('T, km')
ax.set_ylabel('N, km')
ax.set_zlabel('R, km')
plt.legend() 
plt.axis('equal')


# control history 
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(t[:prob.n_time-1], sol["v"][:,0], label='T')
ax.plot(t[:prob.n_time-1], -sol["v"][:,1], label='N')
ax.plot(t[:prob.n_time-1], -sol["v"][:,2], label='R')
ax.set_xlabel('time, TU')
ax.set_ylabel('control')
plt.legend()

plt.show() 
print("done!!")


