import matplotlib.pyplot as plt

from problem_class import *
from scvx_scp import *
from ocp import *
from postprocess import *

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)

# KOZ of 1km for position and 0.001 km/s for velocity
dim = np.array([1/LU, 1/LU, 1/LU, 0.001/LU*TU, 0.001/LU*TU, 0.001/LU*TU])

p_trans = SCVX_OCP(period=t[-1],initial_conditions_target=target_traj[0], N_BRS=10, iter_max=100, koz_dim=dim,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=180,tf=1,mu0=None,muf=None,control=True)

p_trans.μf = np.array([-5/LU,-2/LU,-1/LU,0,0,0])
p_trans.μ0 = np.array([10/LU,6/LU,20/LU,0,0,0])

p_trans.con_list["BRS"] = True
# think about what I could print/plot to check the BRS constraints
# think about initial/final conditions with which it would be easy to check as well

# Seems to work now, try different initial conditions, more or less close to the area where linear approx of the dynamics hold
# Maybe for each position of the final trajectory, compute the distance to the unsafe ellipsoids to make sure we stay outside
# + plot them for a couple of the position
# Make sure the shapes of the ellipsoids are changing as well

p_trans.load_traj_data(fname)
p_trans.linearize_trans()



sol_0 = ocp_cvx(p_trans)
print(sol_0["control_cost"])
plot_chaser_traj_lvlh(sol_0["mu"],LU)
# plt.show()
μref = sol_0["mu"]
p_trans.s_ref = μref
p_trans.get_unsafe_ellipsoids()

prob, log = scvx_star(p_trans, sol_0, μref, fname)
traj = prob.s_ref
print(log["f0"])
# print(np.asarray(log["a"])[-1])

plt.figure()
plt.plot(np.asarray(log["a"])[-1,:,0], label='T') # same units as acceleration m/s^2
plt.plot(np.asarray(log["a"])[-1,:,1], label='N')
plt.plot(np.asarray(log["a"])[-1,:,2], label='R')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_ellipse_3D(p_trans.inv_PP[85,0,:3,:3], ax, LU, TU, label='Final KOZ', color='b', type='pos')
plot_ellipse_3D(p_trans.inv_Pf[:3,:3], ax, LU, TU, label='Final KOZ', color='r', type='pos')
plot_chaser_traj_lvlh_scvx(traj, ax, LU)
ax.scatter(traj[85,0]*LU, -traj[85,1]*LU, -traj[85,2]*LU, color='y', marker='*')
plt.legend()
plt.show()

