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

p_trans.μf = np.array([1.1/LU,0,0,0,0,0])
p_trans.μ0 = np.array([10/LU,0,50/LU,0,0,0])

# think about what I could print/plot to check the BRS constraints
# think about initial/final conditions with which it would be easy to check as well

p_trans.load_traj_data(fname)
p_trans.linearize_trans()



sol_0 = ocp_cvx(p_trans)
plot_chaser_traj_lvlh(sol_0["mu"],LU)
# plt.show()
μref = sol_0["mu"]
# print(μref.shape)

prob, log = scvx_star(p_trans, sol_0, μref, fname)
traj = prob.s_ref
print(traj.shape)
plot_chaser_traj_lvlh(traj,LU)
plt.show()

