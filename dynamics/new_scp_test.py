import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import sys
import os

from linear_dynamics_LVLH import *
from problem_class import *
from ocp import *
from postprocess import *
from new_scvx import *


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)

# KOZ of 1km for position and 0.001 km/s for velocity
dim = np.array([1/LU, 1/LU, 1/LU, 0.001/LU*TU, 0.001/LU*TU, 0.001/LU*TU])

p_trans = SCVX_OCP(period=t[-1],initial_conditions_target=target_traj[0], N_BRS=50, iter_max=100, koz_dim=dim,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=180,tf=1,mu0=None,muf=None,control=True)

p_trans.μ0 = np.array([ -20,    0, 4,   0.0039216,    0, -0.0012745])/LU
p_trans.μf = np.array([ -12,    0, 1.4, 0.003858,     0, 0])/LU

# LVLH [i,j,k] = [T, -N, -R]

p_trans.con_list["BRS"] = False

# loading the data in the problem and linearizing the dynamics (i.e. computing state transition matrix)
p_trans.load_traj_data(fname)
p_trans.linearize_trans()

sol_0 = ocp_cvx_scvx(p_trans)
p_trans.sol_0 = sol_0
print(sol_0["control_cost"])

# plotting the solution given by the basic ocp algorithm
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_chaser_traj_lvlh_scvx(sol_0["s"],ax,LU , 'r')
plot_ellipse_3D(p_trans.inv_Pf[:3,:3], ax, LU, TU, label='Final KOZ', color='r', type='pos')
plt.legend()
plt.show()

sref = sol_0["s"]
aref = sol_0["a"]
p_trans.s_ref = sref
p_trans.a_ref = aref
# given N_BRS and the final keep-out-zone we can compute the unsafe ellipsoid for each time step ahead of time
# to avoid computing them every time we want to solve the optimization control problem
p_trans.get_unsafe_ellipsoids()

# solving the sequential convex programming
sol, log = solve_scvx(p_trans)
traj = sol["s"] # in the LVLH frame
controls = sol["a"]




