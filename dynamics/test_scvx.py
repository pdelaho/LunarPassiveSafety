import matplotlib.pyplot as plt
from scipy.integrate import odeint

from problem_class import *
from scvx_scp import *
from ocp import *
from postprocess import *
from linear_dynamics_LVLH import *
from dynamics_translation import *

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
fname = root_folder + "/dynamics/data/9_2_S_Halo.json"
t, target_traj, mu, LU, TU = load_traj_data(fname)
# print(LU/TU**2)
# print(t.size)

# KOZ of 1km for position and 0.001 km/s for velocity
dim = np.array([1/LU, 1/LU, 1/LU, 0.001/LU*TU, 0.001/LU*TU, 0.001/LU*TU])

p_trans = SCVX_OCP(period=t[-1],initial_conditions_target=target_traj[0], N_BRS=50, iter_max=100, koz_dim=dim,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=180,tf=1,mu0=None,muf=None,control=True)

# Array of way points to test my scp program
wyp   = np.array([[ -20,    0, 4,   0.0039216,    0, -0.0012745,   ],
                       [ -12,    0, 1.4, 0.003858,     0, 0,           ],
                       [ -1.120, 0, 1.4, 0,            0, 0,           ], 
                       [ -1.12,  0, 1.4, 0.00060,      0, -0.0005185,  ],
                       [ 0.5,    0, 0.0,  -0.00027778,  0, -0.00027778, ],
                       [ 0,      0, -0.5,  -0.00027778,  0, 0.00027778, ],
                       [ -0.5,   0, 0.0,   0.000277778, 0, 0.00027778,  ],
                       ]) / LU

p_trans.μ0 = np.array([ -20,    0, 4,   0.0039216,    0, -0.0012745])/LU
p_trans.μf = np.array([ -12,    0, 1.4, 0.003858,     0, 0])/LU # initial situation [10/LU,6/LU,20/LU,0,0,0], in the lvlh frame
# LVLH [i,j,k] = [T, -N, -R]

p_trans.con_list["BRS"] = False
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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_chaser_traj_lvlh_scvx(sol_0["mu"],ax,LU , 'r')
plot_ellipse_3D(p_trans.inv_Pf[:3,:3], ax, LU, TU, label='Final KOZ', color='r', type='pos')
plt.legend()

# plt.show()
μref = sol_0["mu"]
p_trans.s_ref = μref
p_trans.get_unsafe_ellipsoids()

prob, log = scvx_star(p_trans, sol_0, μref, fname)
traj = prob.s_ref # in the LVLH frame
print(log["f0"])
# print(np.asarray(log["a"])[-1])

plt.figure()
plt.plot(np.asarray(log["a"])[-1,:,0], label='T') # same units as acceleration m/s^2
plt.plot(np.asarray(log["a"])[-1,:,1], label='N')
plt.plot(np.asarray(log["a"])[-1,:,2], label='R')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# plot_ellipse_3D(p_trans.inv_PP[85,0,:3,:3], ax, LU, TU, label='Final KOZ', color='b', type='pos')
plot_ellipse_3D(p_trans.inv_Pf[:3,:3], ax, LU, TU, label='Final KOZ', color='r', type='pos')
plot_chaser_traj_lvlh_scvx(traj, ax, LU, 'r') # transfo to RTN happening in there
# ax.scatter(traj[85,0]*LU, -traj[85,1]*LU, -traj[85,2]*LU, color='y', marker='*')
# plt.legend()
# plt.show()

# now to check that all the constraints are respected, trajectory for every point of the final solution + final KOZ
# print(traj.shape)
for i in range(0,traj.shape[0],10):
    # we take a point of the traj and use non linear dynamics (with no control actions) to propagate it for the N_BRS steps
    # first step is to put the initial point in the Moon frame
    initial_point = lvlh_to_synodic(traj[i,:],prob.target_traj[i],mu)
    t_simulation = prob.time_hrz[i:i+prob.N_BRS]
    # print(t_simulation[-1]-t_simulation[0])
    indiv_traj_moon = odeint(propagator_absolute,initial_point,t_simulation,args=(mu,))
    indiv_traj_lvlh = np.empty_like(indiv_traj_moon)
    # print(indiv_traj_moon.shape)
    for j in range(indiv_traj_moon.shape[0]):
        indiv_traj_lvlh[j] = synodic_to_lvlh(indiv_traj_moon[j],prob.target_traj[i+j], mu)
    plot_chaser_traj_lvlh_check(indiv_traj_lvlh, ax, LU, 'c', 1.5)

# we also might wanna compare to the trajectory without any controls (to see if controls needed to avoid the KOZ)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
initial_point_moon = lvlh_to_synodic(traj[0,:],prob.target_traj[0,:],mu)
t_simulation = prob.time_hrz[:prob.n_time]
traj_moon_noControls = odeint(propagator_absolute,initial_point_moon,t_simulation,args=(mu,))
# print(traj_moon_noControls.shape)
# now put everything back in the moon frame
traj_lvlh_noControls = np.empty_like(traj_moon_noControls)
for i in range(traj_moon_noControls.shape[0]):
    traj_lvlh_noControls[i] = synodic_to_lvlh(traj_moon_noControls[i,:],prob.target_traj[i,:],mu)
    
# print(traj_lvlh_noControls.shape)

ax.plot(traj_lvlh_noControls[:,0]*LU, -traj_lvlh_noControls[:,1]*LU, -traj_lvlh_noControls[:,2]*LU, label='Trajectory without using controls',color='r',linewidth=1)
ax.scatter(traj_lvlh_noControls[0,0]*LU,-traj_lvlh_noControls[0,1]*LU,-traj_lvlh_noControls[0,2]*LU,label='Start')
ax.scatter(traj_lvlh_noControls[-1,0]*LU,-traj_lvlh_noControls[-1,1]*LU,-traj_lvlh_noControls[-1,2]*LU,label='End')
plot_ellipse_3D(p_trans.inv_Pf[:3,:3], ax, LU, TU, label='Final KOZ', color='r', type='pos')
ax.axis('equal')
# LVLH [i,j,k] = [T, -N, -R]
ax.set_xlabel('T [km]')
ax.set_ylabel('N [km]')
ax.set_zlabel('R [km]')
ax.legend()
plt.title("Chaser's trajectory in the LVLH frame without controls")
plt.grid()
plt.legend()
# plt.show()

# make 2D plots in the RT/RN/TN frames

fig = plt.figure()
ax = fig.add_subplot()
plt.title("Trajectory in the RT plane")
ax.set_xlabel('R [km]')
ax.set_ylabel('T [km]')
ax.plot(-traj[:,2]*LU, traj[:,0]*LU, label='Trajectory', color='r', linewidth=1)
ax.scatter(0,0, color='y',marker='*',label="Target's position")
plt.legend()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.title("Trajectory in the RN plane")
ax.set_xlabel('R [km]')
ax.set_ylabel('N [km]')
ax.plot(-traj[:,2]*LU, -traj[:,1]*LU, label='Trajectory', color='r', linewidth=1)
ax.scatter(0,0, color='y',marker='*',label="Target's position")
plt.legend()

# This was printing the trajectory of the target instead of the natural dynamics of the chaser given the initial condition

# fig = plt.figure()
# ax = fig.add_subplot()
# plt.title("Trajectory in the TN plane")
# ax.set_xlabel('T [km]')
# ax.set_ylabel('N [km]')
# ax.plot(traj[:,0]*LU, -traj[:,1]*LU, label='Trajectory', color='r', linewidth=1)
# ax.scatter(0,0, color='y',marker='*',label="Target's position")
# plt.legend()

plt.show()