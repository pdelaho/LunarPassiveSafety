import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import random

from problem_class import *
from ocp import *
from postprocess import *

L2x = 1.15568217 # nd, position of the L2 point

x0_bary = 9.2280005557282274E-1 # nd, initial conditions from Franzini's article
y0_bary = 1.6386233489716853E-28 # nd
z0_bary = -2.1575768509057866E-1 # nd
vx0_bary = 4.4327633188679963E-13 # nd
vy0_bary = 	1.2826547451754347E-1 # nd
vz0_bary = 2.4299327620081873E-12 # nd
T = 1.8036720655626510E+0 # nd
initial_conditions_bary = np.asarray([x0_bary, y0_bary, z0_bary, vx0_bary, vy0_bary, vz0_bary])
initial_conditions_synodic = bary_to_synodic(initial_conditions_bary, mu=1.215e-2)

p_trans = CR3BP_RPOD_OCP(
    period=T, initial_conditions_target=initial_conditions_synodic, iter_max=15,
    mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
    n_time=1000,nx=6,nu=3,M0=180,tf=0.07,mu0=None,muf=None,control=True
)

# Set-up of the dynamics
p_trans.get_traj_ref(p_trans.n_time)
p_trans.linearize_trans()

# Define the initial conditions for the chaser spacecraft
# For now initial conditions are random
distance_to_target_km = 10 # km
distance_to_target = distance_to_target_km/p_trans.LU # nd

rho_x0_lvlh = rand()*distance_to_target*random.choice([1,-1])
rho_y0_lvlh = rand()*np.sqrt(distance_to_target**2 - rho_x0_lvlh**2)*random.choice([1,-1])
rho_z0_lvlh = np.sqrt(distance_to_target**2 - rho_x0_lvlh**2 - rho_y0_lvlh**2)*random.choice([1,-1])

velocity_rel_target_km = 0 # km/s
velocity_rel_target = velocity_rel_target_km/p_trans.LU*p_trans.TU # nd

rho_vx0_lvlh = rand()*velocity_rel_target
rho_vy0_lvlh = rand()*np.sqrt(velocity_rel_target**2 - rho_vx0_lvlh**2)
rho_vz0_lvlh = np.sqrt(velocity_rel_target**2 - rho_vx0_lvlh**2 - rho_vy0_lvlh**2)

p_trans.μ0 = np.asarray([rho_x0_lvlh, rho_y0_lvlh, rho_z0_lvlh, rho_vx0_lvlh, rho_vy0_lvlh, rho_vz0_lvlh])
p_trans.get_chaser_nonlin_traj()

# Setting final conditions of the ocp
p_trans.μf = p_trans.chaser_nonlin_traj[-1,:]

sol = ocp_cvx(p_trans)
chaser_traj = sol["mu"]
l_opt = sol["l"]
a_opt = sol["v"]

# Plotting the orbit of the target
plot_target_traj_syn(p_trans.target_traj, L2x, p_trans.mu)
plt.show()

# Plotting the trajectory of the chaser (result of optimization)
plot_chaser_traj_lvlh(chaser_traj)
plt.show()

# Plotting the trajectory of th chaser (propagation using non-linear dynamics)
plot_chaser_traj_lvlh(p_trans.chaser_nonlin_traj)
plt.show()

plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,[np.linalg.norm(l_opt[i,:3])*p_trans.LU*1e3 for i in range(l_opt.shape[0])])
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\vec{l_{pos}}$|| [m]')
plt.title('Norm of the position part of the slack variable')
plt.show()

plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,[np.linalg.norm(l_opt[i,3:6])*p_trans.LU*1e3/p_trans.TU for i in range(l_opt.shape[0])])
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\vec{l_{vel}}$|| [m/s]')
plt.title('Norm of the velocity part of the slack variable')
plt.show()

plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,a_opt[:,0]*p_trans.LU/(p_trans.TU**2), label='X component',linewidth=1)
plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,a_opt[:,1]*p_trans.LU/(p_trans.TU**2), label='Y component',linewidth=1)
plt.plot(p_trans.time_hrz[1:]*p_trans.TU/3600,a_opt[:,2]*p_trans.LU/(p_trans.TU**2), label='Z component',linewidth=1)
plt.legend()
plt.xlabel('Time [hours]')
plt.ylabel(r'Components of the control input [m/$s^2$]')
plt.title('Control inputs over time')
plt.show()

error_pos, error_vel = analysis(p_trans.chaser_nonlin_traj,chaser_traj,p_trans.n_time)
plt.plot(p_trans.time_hrz*p_trans.TU/3600,error_pos*p_trans.LU*1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\rho - \hat{\rho}$|| [m]')
plt.title('Norm of the difference on the relative position of the chaser spacecraft in the lvlh frame')
plt.show()

plt.plot(p_trans.time_hrz*p_trans.TU/3600,error_vel*p_trans.LU*1e3/p_trans.TU, linewidth=1)
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\dot{\rho} - \dot{\hat{\rho}}$|| [m/s]')
plt.title('Norm of the difference on the relative velocity of the chaser spacecraft in the lvlh frame')
plt.show()