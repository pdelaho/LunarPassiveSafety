import numpy as np
import numpy.linalg as la
import scipy as sc 
import cvxpy as cp
import matplotlib.pyplot as plt
import csv
import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la
from numpy import cross
from numpy.random import rand
import matplotlib.pyplot as plt
import csv
import scipy as sc
import os
import sys


from linear_dynamics_LVLH import matrix_dynamics
from useful_small_functions import get_phi


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
# fname = root_folder + "\\dynamics\\data\\target_chaser_trajectories.csv"
fname = root_folder + "\\data\\target_chaser_trajectories.csv"
# maybe use the json file for more accurate data?
# fname = "C:\\Users\\pauli\\Documents\\SLAB\\LunarPassiveSafety\\dynamics\\data\\traj_target_chaser.csv"

file = open(fname, "r")
reader = csv.reader(file)
data_target = []
data_chaser = []
time_steps  = []
for lines in reader:
    if lines != []:
        time_steps.append(lines[0])
        data_target.append(lines[1:7])
        data_chaser.append(lines[7:])
file.close()

# Data for the Earth-Moon system
r12 = 384400 # km
μ = 1.215e-2 # no unit, mass parameter of the system
TU = 1 / 2.661699e-6 # s, inverse of the relative angular frequency between the two primary attractors
L1x = 0.83691513 # nd, position of the L1 point along the x direction
L2x = 1.15568217 # nd, position of the L2 point along the x direction

nx = 6 # dimension of the state vector
nu = 3 # dimension of the control input vector
# n_time = len(data_target) # number of time steps
n_time = 250 # number of time steps
s_0, s_f = data_chaser[0], data_chaser[n_time - 1]

# Control input matrix
B = np.zeros((6, 3))
B[3:, :] = np.eye(3)

# Reformated data
target = np.zeros((len(data_target), len(data_target[0])))
chaser = np.zeros((len(data_chaser), len(data_chaser[0])))
time = np.zeros((len(time_steps),))
for i in range(n_time):
    target[i] = data_target[i][:]
    chaser[i] = data_chaser[i][:]
    time[i] = time_steps[i]

# normalized vbariables, working with adimensionalized variables for now
s = cp.Variable((n_time,     nx))
l = cp.Variable((n_time - 1, nx))  # slack variable to prevent artificial infeasibility
a = cp.Variable((n_time - 1, nu))

# 1st possible optimization problem
con = []
cost = 0
con += [s[0] == s_0]
# con += [s[i+1] == sc.linalg.expm((target[i+1][0]-target[i][0])*matrix_dynamics(target[i,1:],0,mu)) @ s[i] + l[i] for i in range(n_time-1)]
con += [s[i + 1] == get_phi(time[i + 1] - time[i], matrix_dynamics(target[i][:], μ)) @ s[i] \
                  + B @ a[i] + l[i] for i in range(n_time - 1)]
# con += [s[i+1] == matrix_dynamics(target[i+1][0]-target[i][0],target[i,1:],mu) @ s[i] + l[i] for i in range(n_time-1)]
# con += [s[i+1] == propagator_relative_with_matrix(target[i,1:],target[i+1][0]-target[i][0],mu) @ s[i] + l[i] for i in range(n_time-1)]
con += [s[-1] == s_f]

J = cp.sum(cp.norm(a, 2, axis=0))
cost += J
cost += cp.sum(cp.norm(l, 2, axis=0)) * 1e6   # slack variable penalty

p = cp.Problem(cp.Minimize(cost), con)
p.solve(solver=cp.MOSEK, verbose=False) # MOSEK, CLARABEL
s_opt  = s.value
l_opt  = l.value
a_opt  = a.value
J_opt  = J.value
print(J_opt)

norm_l_dist = np.zeros(n_time - 1)
norm_l_vel  = np.zeros(n_time - 1)
for l in range(n_time - 1):
    norm_l_dist[l] = la.norm(l_opt[l,  :3])
    norm_l_vel[l]  = la.norm(l_opt[l, 3:6])
    
# Plotting the norm of the slack variable (that should be close to 0 since same dynamics)
plt.figure()
plt.plot(time[1:n_time] * TU / 3600, norm_l_dist * r12 * 1e3)
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\vec{l}$|| [m]')
plt.title('Norm of the slack variable position part')

plt.figure()
# plt.plot(target[1:n_time, 0] * TU / 3600, norm_l_dist * r12 * 1e3 / TU)
plt.plot(time[1:n_time] * TU / 3600, norm_l_vel * r12 * 1e3 / TU)
plt.xlabel('Time [hours]')
plt.ylabel(r'||$\vec{l}$|| [m/s]')
plt.title('Norm of the slack variable velocity part')

# Plotting the control inputs over time
plt.figure()
plt.plot(time[1:n_time] * TU / 3600, a_opt[:, 0] * r12 * 1e3 / TU**2, label='X component')
plt.plot(time[1:n_time] * TU / 3600, a_opt[:, 1] * r12 * 1e3 / TU**2, label='Y component')
plt.plot(time[1:n_time] * TU / 3600, a_opt[:, 2] * r12 * 1e3 / TU**2, label='Z component')
plt.xlabel('Time [hours]')
plt.ylabel(r'Components of the control input a [m/s^2]')
plt.title('Control inputs')
plt.legend()

# Plotting the optimized trajectory
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(s_opt[:, 0], s_opt[:, 1], s_opt[:, 2], color='r', label="Chaser's trajectory")
ax.scatter(0, 0, 0, label='Target')
ax.axis('equal')
ax.set_xlabel('X [nd]')
ax.set_ylabel('Y [nd]')
ax.set_zlabel('Z [nd]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame")
plt.grid()

plt.show()