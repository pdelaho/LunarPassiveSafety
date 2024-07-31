import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import random
import sys
import os

from problem_class import *
from ocp import *
from postprocess import *
from safe_set import *

L2x = 1.15568217 # nd, position of the L2 point

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)


## Tests for the BRS using ellipsoids as the KOZ

# Defining the unsafe ellipsoid (6D, 3D for position and 3D for velocity)
rx = 10 # km
rx_ad = 10/LU
rv = 0.1 # km/s
rv_ad = rv/LU*TU
# volume_tot = 1/np.sqrt(np.pi*6) * (2*np.pi*math.e/6)**(3) * rx_ad**3 * rv_ad**3
# print(volume_tot)
Pf = np.diag([rx_ad**2, rx_ad**2, rx_ad**2, rv_ad**2, rv_ad**2, rv_ad**2])
inv_Pf = np.linalg.inv(Pf)

# Defining the final time step at which the chaser should be oustide the KOZ
final_time_step = 1000

# N = 100
# inv_PP = passive_safe_ellipsoid(p_trans, N, inv_Pf) # computing the unsafe ellipsoids
# eig_values, _ = np.linalg.eig(inv_PP[0])
# volume_tot = 1/np.sqrt(np.pi*6) * (2*np.pi*math.e/6)**(3) * eig_values[0] * eig_values[1] * eig_values[2] * eig_values[3] * eig_values[4] * eig_values[5]
# print(volume_tot)
# print(np.trace((inv_PP[1,:3,:3]-inv_PP[0,:3,:3]) @ (inv_PP[1,:3,:3]-inv_PP[0,:3,:3]).T), np.trace((inv_PP[1,:3,:3]-inv_PP[2,:3,:3]) @ (inv_PP[1,:3,:3]-inv_PP[2,:3,:3]).T), np.trace((inv_PP[1,:3,:3]-inv_PP[-1,:3,:3]) @ (inv_PP[1,:3,:3]-inv_PP[-1,:3,:3]).T))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# plot_ellipse_3D(inv_Pf[3:6,3:6], ax, LU)

# Change the plotting function of the ellipsoid to plot in 3D and not 2D
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for k in range(N):
# plot_ellipse_3D(inv_PP[0,0:3,0:3], ax, LU)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# plot_ellipse_3D(inv_PP[-1,3:6,3:6], ax, LU)

plt.show()

p_trans2 = CR3BP_RPOD_OCP(
    period=t[-1], initial_conditions_target=target_traj[0], iter_max=15,
    mu=mu,LU=LU,mean_motion=2.661699e-6,
    n_time=final_time_step+1,nx=6,nu=3,M0=180,tf=0.5,mu0=None,muf=None,control=False
)
p_trans2.load_traj_data(fname)
p_trans2.linearize_trans()

N = final_time_step
inv_PP = passive_safe_ellipsoid(p_trans2, N, inv_Pf, final_time_step) # computing the unsafe ellipsoids

# Generating a random vector outside the unsafe ellipsoid
x_out = generate_outside_ellipsoid(inv_PP[-1], np.asarray([0,0,0,0,0,0]))
print(x_out, x_out @ inv_PP[-1] @ x_out.T)

x_in = generate_inside_ellipsoid(inv_PP[-1], np.asarray([0,0,0,0,0,0]))
print(x_in, x_in @ inv_PP[-1] @ x_in.T)

p_trans2.μ0 = x_out
# p_trans2.μ0 = x_in

p_trans2.get_chaser_nonlin_traj()

sol2 = ocp_cvx(p_trans2)
chaser_traj = sol2["mu"]
l_opt = sol2["l"]
a_opt = sol2["v"]

fig = plt.figure()
plt.plot(p_trans2.time_hrz[1:final_time_step+1]*TU/3600,a_opt[:final_time_step,0]*LU/(TU**2), label='T',linewidth=1)
plt.plot(p_trans2.time_hrz[1:final_time_step+1]*TU/3600,-a_opt[:final_time_step,1]*LU/(TU**2), label='N',linewidth=1)
plt.plot(p_trans2.time_hrz[1:final_time_step+1]*TU/3600,-a_opt[:final_time_step,2]*LU/(TU**2), label='R',linewidth=1)
plt.legend()
plt.xlabel('Time [hours]')
plt.ylabel(r'Components of the control input [m/$s^2$]')
plt.title('Control inputs over time')
plt.show()

print(chaser_traj[final_time_step] @ inv_PP[-1] @ chaser_traj[final_time_step].T)
print(p_trans2.chaser_nonlin_traj[final_time_step] @ inv_PP[-1] @ p_trans2.chaser_nonlin_traj[final_time_step].T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_ellipse_3D(inv_Pf[3:6,3:6], ax, LU, TU, 'Final KOZ', 'b', 'vel')
plot_ellipse_3D(inv_PP[-1,3:6,3:6], ax, LU, TU, 'Initial unsafe ellipsoid', 'r', 'vel')
plt.show()