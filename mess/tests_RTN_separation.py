import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from dynamics.problem_class import *
from dynamics.ocp import *
from dynamics.postprocess import *
from dynamics.dynamics_translation import *
from dynamics.useful_small_functions import *
from dynamics.linear_dynamics_LVLH import *

L2x = 1.15568217 # nd, position of the L2 point

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)

p_trans = CR3BP_RPOD_OCP(
    period=t[-1], initial_conditions_target=target_traj[0], iter_max=15,
    mu=mu,LU=LU,mean_motion=2.661699e-6,
    n_time=3000,nx=6,nu=3,M0=180,tf=0.5,mu0=None,muf=None,control=True
)

# Set-up of the dynamics
# p_trans.get_traj_ref(p_trans.n_time)
p_trans.load_traj_data(fname)
p_trans.linearize_trans()

# Define the initial conditions for the chaser spacecraft
# LVLH [i,j,k] = [T, -N, -R]
init_cond = np.asarray([0,10,0,0,0,0])/LU # nd

p_trans.μ0 = init_cond
p_trans.get_chaser_nonlin_traj()

p_trans.μf = np.asarray([0,0,0,0,0,0]) # p_trans.μf = p_trans.chaser_nonlin_traj[-1]

sol = ocp_cvx(p_trans)
chaser_traj = sol["mu"]
l_opt = sol["l"]
a_opt = sol["v"]

# Plotting the orbit of the target
plot_target_traj_syn(p_trans.target_traj, L2x, p_trans.mu)

# Plotting the trajectory of the chaser (result of optimization)
plot_chaser_traj_lvlh(chaser_traj,p_trans.LU)

# plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,[np.linalg.norm(l_opt[i,:3])*p_trans.LU*1e3 for i in range(l_opt.shape[0])])
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$\vec{l_{pos}}$|| [m]')
# plt.title('Norm of the position part of the slack variable')
# plt.show()

# plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,[np.linalg.norm(l_opt[i,3:6])*p_trans.LU*1e3/p_trans.TU for i in range(l_opt.shape[0])])
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$\vec{l_{vel}}$|| [m/s]')
# plt.title('Norm of the velocity part of the slack variable')
# plt.show()

fig = plt.figure()
plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,a_opt[:,0]*p_trans.LU/(p_trans.TU**2), label='T',linewidth=1)
plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,-a_opt[:,1]*p_trans.LU/(p_trans.TU**2), label='N',linewidth=1)
plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,-a_opt[:,2]*p_trans.LU/(p_trans.TU**2), label='R',linewidth=1)
plt.legend()
plt.xlabel('Time [hours]')
plt.ylabel(r'Components of the control input [m/$s^2$]')
plt.title('Control inputs over time')

# plt.show()

# error_pos, error_vel = analysis(p_trans.chaser_nonlin_traj,chaser_traj,p_trans.n_time)
# plt.plot(p_trans.time_hrz*p_trans.TU/3600,error_pos*p_trans.LU*1e3)
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$\rho - \hat{\rho}$|| [m]')
# plt.title('Norm of the difference on the relative position of the chaser spacecraft in the lvlh frame')
# plt.show()

# plt.plot(p_trans.time_hrz*p_trans.TU/3600,error_vel*p_trans.LU*1e3/p_trans.TU, linewidth=1)
# plt.xlabel('Time [hours]')
# plt.ylabel(r'||$\dot{\rho} - \dot{\hat{\rho}}$|| [m/s]')
# plt.title('Norm of the difference on the relative velocity of the chaser spacecraft in the lvlh frame')
# plt.show()



## Test of the natural (nonlinear) dynamics in the context of R/T/N separation (to verify our intuition)

# LVLH [i,j,k] = [T, -N, -R]
init_cond_lvlh = np.asarray([0,0,10,0,0,0])/LU # to adimensionalize the distances (but not velocities!)
init_cond_syn = lvlh_to_synodic(init_cond_lvlh,target_traj[0],p_trans.mu)

chaser_traj_syn = odeint(dynamics_synodic,init_cond_syn,t[:200],args=(p_trans.mu,))
trajectories = odeint(propagator_relative,np.concatenate((target_traj[0],init_cond_lvlh)),t[:200],args=(p_trans.mu,))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r', label="Target's orbit")
ax.scatter(0, 0, 0, label='Moon')
ax.scatter(L2x-(1-mu), 0, 0, label='L2')
ax.scatter(target_traj[0,0],target_traj[0,1],target_traj[0,2], label='Start')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Target's orbit in the synodic frame")
plt.grid()
# plt.show()

chaser_traj_lvlh = np.empty_like(chaser_traj_syn)
for i in range(chaser_traj_lvlh.shape[0]):
    chaser_traj_lvlh[i] = synodic_to_lvlh(chaser_traj_syn[i],target_traj[i],p_trans.mu)

plot_chaser_traj_lvlh(chaser_traj_lvlh,LU)
plot_chaser_traj_lvlh(trajectories[:,6:12],LU)
plt.show()

# Check again of linear/non-linear dynamics still being close enough for the approximation to hold
# norm_error = np.empty_like(chaser_traj_lvlh[:,0])
# for i in range(chaser_traj_lvlh.shape[0]):
#     norm_error[i] = np.linalg.norm(chaser_traj_lvlh[i,:3] - trajectories[i,6:9])
    
# plt.plot(t[:200]*TU/3600,norm_error*LU*1e3)
# plt.show()