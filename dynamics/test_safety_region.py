import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from problem_class import *
from ocp import *
from postprocess import *
from safe_set import *

# TO DO: CLEAN UP THIS MESS AND SEE WHAT WERE THE ACTUAL RELEVANT TESTS FOR THE UNSAFE ELLIPSOIDS COMPUTATION

L2x = 1.15568217 # nd, position of the L2 point

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)


## Tests for the part where we compute the unsafe ellipsoids

# Defining the unsafe ellipsoid (6D, 3D for position and 3D for velocity)
rx = 10 # km
rx_ad = rx/LU
rv = 0.001 # km/s
rv_ad = rv/LU*TU
Pf = np.diag([rx_ad**2, rx_ad**2, rx_ad**2, rv_ad**2, rv_ad**2, rv_ad**2])
inv_Pf = np.linalg.inv(Pf)

# Defining the final time step at which the chaser should be oustide the KOZ
final_time_step = 1000

# Initialieing the problem with the data from a json file
p_trans = CR3BP_RPOD_OCP(
    period=t[-1], initial_conditions_target=target_traj[0], iter_max=15,
    mu=mu,LU=LU,mean_motion=2.661699e-6,
    n_time=final_time_step+1,nx=6,nu=3,M0=180,tf=0.5,mu0=None,muf=None,control=False
)
p_trans.load_traj_data(fname)
p_trans.linearize_trans()


N = final_time_step
inv_PP = passive_safe_ellipsoid(p_trans, N, inv_Pf, final_time_step)

# print(volume_ellipsoid(inv_Pf, LU, TU),volume_ellipsoid(inv_PP[-1], LU, TU))

# x_out = np.asarray([1*1e-5,0,3*1e-5,0,0,0])
# x_out = np.asarray([0,0,2.48*1e-5,0,0,0])
# x_out = np.asarray([3*1e-5,0,1e-5,0,0,0])
# x_out = np.asarray([0,0,0,1.5*1e-3,0,5*1e-4])
x_out = np.asarray([0,0,0,2.5*1e-4,0,2.5*1e-4])

print(x_out @ inv_PP[-1] @ x_out.T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_ellipse_3D(inv_Pf[:3,:3], ax, LU, TU, 'Final KOZ', 'b', 'pos')
plot_ellipse_3D(inv_PP[-1,:3,:3], ax, LU, TU, 'Initial unsafe ellipsoid', 'r', 'pos')
ax.scatter(x_out[0],x_out[1],x_out[2], label='Start', color='k')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_ellipse_3D(inv_Pf[3:6,3:6], ax, LU, TU, 'Final KOZ', 'b', 'vel')
plot_ellipse_3D(inv_PP[-1,3:6,3:6], ax, LU, TU, 'Initial unsafe ellipsoid', 'r', 'vel')
ax.scatter(x_out[3],x_out[4],x_out[5], label='Start', color='k')
plt.legend()


p_trans.μ0 = x_out

p_trans.get_chaser_nonlin_traj()

sol2 = ocp_cvx(p_trans)
chaser_traj = sol2["mu"]
l_opt = sol2["l"]
a_opt = sol2["v"]

print(chaser_traj[final_time_step] @ inv_Pf @ chaser_traj[final_time_step].T)
print(p_trans.chaser_nonlin_traj[final_time_step] @ inv_Pf @ p_trans.chaser_nonlin_traj[final_time_step].T)

closest_ellipsoids, indices = extract_closest_ellipsoid(x_out, inv_PP, 1)
closest_ellipsoids = np.asarray(closest_ellipsoids)
print(indices)

h = convexify_safety_constraint(x_out, closest_ellipsoids, 1)
# print(h)

# create x,y
xx, yy = np.meshgrid(range(0,40), range(-50,60))
# xx, yy = np.meshgrid(range(-2,2), range(-2,2))
xx = np.asarray(xx)*1e-6
yy = np.asarray(yy)*1e-6

# calculate corresponding z
z = (-h[0] * xx - h[1] * yy + 1) * 1. / h[2]

# plot the surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, z, alpha=1, label='Hyperplane')
# plot_ellipse_3D(inv_Pf[:3,:3], ax, LU, TU, 'Final KOZ', 'b', 'pos')
plot_ellipse_3D(closest_ellipsoids[0,:3,:3], ax, LU, TU, 'Closest ellipsoid', 'r', 'pos')
ax.scatter(x_out[0], x_out[1], x_out[2], c='k')
ax.axis('equal')
plt.legend()

xx, yy = np.meshgrid(range(-30,30), range(-30,30))
xx = np.asarray(xx)*1e-4
yy = np.asarray(yy)*1e-4

# calculate corresponding z
z = (-h[3] * xx - h[4] * yy + 1) * 1. /h[5]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, z, alpha=1, label='Hyperplane')
# plot_ellipse_3D(inv_Pf[3:6,3:6], ax, LU, TU, 'Final KOZ', 'b', 'vel')
plot_ellipse_3D(closest_ellipsoids[0,3:6,3:6], ax, LU, TU, 'Closest ellipsoid', 'r', 'vel')
ax.scatter(x_out[3], x_out[4], x_out[5], c='k')
ax.axis('equal')
plt.legend()

plt.show()




## Tests for the part where we extract the n closest ellipsoids given a state point

# First trying with ellipsoids and points completely different from the ones we'll have in the RPOD scenario to check
P = np.empty((2,3,3))
P[0] = np.linalg.inv(np.diag([1**2, 2**2, 3**2]))
P[1] = np.linalg.inv(np.diag([3**2, 1**2, 2**2]))

x_test = np.asarray([1,2,3])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# plot_ellipse_3D(P[0], ax, LU, TU, 'Ellipse 1', 'b', 'pos')
# plot_ellipse_3D(P[1], ax, LU, TU, 'Ellipse 2', 'r', 'pos')

# ax.scatter(x_test[0], x_test[1], x_test[2], c='k')
# ax.axis('equal')
# plt.legend()
# plt.show()

closest_ellipsoid, indices = extract_closest_ellipsoid(x_test, P, 1)
# print(indices)

## Tests for the part where we convexify (thanks to a hyperplane) the constraints of staying outside the unsafe ellipsoid

# Keep using the previous example
h = convexify_safety_constraint(x_test, closest_ellipsoid, 1)
# print(h)

# create x,y
xx, yy = np.meshgrid(range(3), range(3))

# calculate corresponding z
z = (-h[0] * xx - h[1] * yy + 1) * 1. /h[2]

# plot the surface
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(xx, yy, z, alpha=1, label='Hyperplane')
# plot_ellipse_3D(P[0], ax, LU, TU, 'Ellipse 1', 'b', 'pos')
# plot_ellipse_3D(P[1], ax, LU, TU, 'Ellipse 2', 'r', 'pos')
# ax.scatter(x_test[0], x_test[1], x_test[2], c='k')
# ax.axis('equal')
# plt.legend()
# plt.show()