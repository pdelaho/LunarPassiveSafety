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
dt = (t[1] - t[0]) * TU   # sec. 
print("traj loaded...")
print(f"mu: {mu} km^3/s^2")   
print(f"LU: {LU} km")
print(f"TU: {TU} s")
print(f"period: {t[-1] * TU / 60 /60 } hrs") 
print(f"dt: {dt} s")

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
prob.μ0 = np.array([-40, 0, 4, 0.005747, 0, 0]) / LU   
prob.μf = np.array([0, 0, 0.5, 0, 0, 0]) / LU
prob.μ0[3:] = prob.μ0[3:] * TU
prob.μf[3:] = prob.μf[3:] * TU
prob.n_time = int(19140 / dt)  

# waypoints 
prob.con_list = {"wyp": True}   

prob.wyp_t = np.array([3480, 5520, 8340, 11040, 13740, 15540, 17340])
prob.wyp   = np.array([[ -20,    0, 4,   0.0039216,    0, -0.0012745,   ],
                       [ -12,    0, 1.4, 0.003858,     0, 0,           ],
                       [ -1.120, 0, 1.4, 0,            0, 0,           ], 
                       [ -1.12,  0, 1.4, 0.00060,      0, -0.0005185,  ],
                       [ 0.5,    0, 0.0,  -0.00027778,  0, -0.00027778, ],
                       [ 0,      0, -0.5,  -0.00027778,  0, 0.00027778, ],
                       [ -0.5,   0, 0.0,   0.000277778, 0, 0.00027778,  ],
                       ]) / LU 
prob.wyp[:,3:] = prob.wyp[:,3:] * TU    
prob.dt = dt

sol = ocp_cvx(prob) 
s  = sol["mu"] 
J   = sol["control_cost"]    
print(f"J (control cost): {J}")

# plot the absolute trajectory (NRHO)
fig = plt.figure(figsize=(10,8)) 
ax  = fig.add_subplot(111, projection='3d') 
ax.plot3D(traj[:,0] , traj[:,1], traj[:,2], 'k') 
ax.scatter3D(traj[0,0] , traj[0,1], traj[0,2], c='r', s=100, label='start')  # initial state is apoapsis
ax.scatter3D(0,0,0, c='orange', s=100, label='Moon')
ax.axis('equal')
ax.set_xlabel('x, LU')
ax.set_ylabel('y, LU') 
ax.set_zlabel('z, LU') 
plt.legend() 


# plot the relative trajecotry (scaled in km) in the RTN frame # LVLH [i,j,k] = [T, -N, -R]
fig = plt.figure(figsize=(10,8)) 
# 3D plot 
# ax  = fig.add_subplot(111, projection='3d') 
# ax.plot3D(s[:,0]*LU, -s[:,1]*LU, -s[:,2]*LU, 'k') 
# ax.scatter3D(s[0,0]*LU,  -s[0,1]*LU,  -s[0,2]*LU,  c='r', s=100, label='start')
# ax.scatter3D(s[-1,0]*LU, -s[-1,1]*LU, -s[-1,2]*LU, c='b', s=100, label='end')
# for i in range(len(prob.wyp)):
#     ax.scatter3D(prob.wyp[i,0]*LU, -prob.wyp[i,1]*LU, -prob.wyp[i,2]*LU, c='orange', s=100, label='waypoint')
# ax.set_xlabel('T, km')
# ax.set_ylabel('N, km')
# ax.set_zlabel('R, km')
# plt.legend() 
# plt.axis('equal')

# 2D plot
ax = fig.add_subplot(111)
ax.plot(s[:,0]*LU, -s[:,2]*LU, 'k')
ax.scatter(s[0,0]*LU, -s[0,2]*LU, c='r', s=20, label='start')
ax.scatter(s[-1,0]*LU, -s[-1,2]*LU, c='b', s=20, label='end')
for i in range(len(prob.wyp)):
    ax.scatter(prob.wyp[i,0]*LU, -prob.wyp[i,2]*LU, c='orange', s=20, label='waypoint')
ax.set_xlabel('T, km')
ax.set_ylabel('R, km')
ax.invert_xaxis()
plt.grid("on")
plt.grid("minor")
plt.legend()


# control history 
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(t[:prob.n_time-1], sol["v"][:,0], label='T')
ax.plot(t[:prob.n_time-1], -sol["v"][:,1], label='N')
ax.plot(t[:prob.n_time-1], -sol["v"][:,2], label='R')
for i in range(len(prob.wyp)):
    idx = int(prob.wyp_t[i] / dt)
    ax.vlines(x=t[idx], ymin=-0.1, ymax=0.1, color='gray', linestyles="dashed", linewidth=0.5, label='waypoint')
ax.set_xlabel('time, TU')
ax.set_ylabel('control')
ymax = np.max(np.max(np.abs(sol["v"])))
ax.set_ylim(-ymax, ymax)
plt.legend()

plt.show() 
print("done!!")


