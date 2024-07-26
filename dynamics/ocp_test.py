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
print(np.shape(traj))

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

# boundary condition (in km, and scaled by LU)
prob.μ0 = np.array([10,0,0,0,0,0])  / LU 
prob.μf = np.array([0,0,0,0,0,0])  / LU
prob.n_time = 200 #len(t)

sol = ocp_cvx(prob) 
mu  = sol["mu"] 


# plot the absolute trajectory (NRHO)
fig = plt.figure(figsize=(10,8)) 
ax  = fig.add_subplot(111, projection='3d') 
ax.plot3D(traj[:,0], traj[:,1], traj[:,2], 'k') 
ax.scatter3D(traj[0,0], traj[0,1], traj[0,2], c='r', s=100, label='start')
plt.legend() 


# plot the relative trajecotry 
# nominal trajectory (sanity check) 
fig = plt.figure(figsize=(10,8)) 
ax  = fig.add_subplot(111, projection='3d') 
ax.plot3D(mu[:,1]*LU, mu[:,2]*LU, mu[:,0]*LU, 'k') 
ax.scatter3D(mu[0,1]*LU,  mu[0,2]*LU,  mu[0,0]*LU,  c='r', s=100, label='start')
ax.scatter3D(mu[-1,1]*LU, mu[-1,2]*LU, mu[-1,0]*LU, c='b', s=100, label='end')
ax.set_xlabel('T, km')
ax.set_ylabel('N, km')
ax.set_zlabel('R, km')
plt.legend() 

plt.show() 


print("done!!")


