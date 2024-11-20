import dataclasses
import numpy as np

from dynamics_linearized import linearize_translation, linearize_translation_scvx
from safe_set import passive_safe_ellipsoid_scvx
from useful_small_functions import load_traj_data
from dynamics_translation import get_chaser_nonlin_traj, get_traj_ref


class CR3BP_RPOD_OCP:
    """
    Optimal Control Problem (OCP) in the context of 3-body problem class
    Default arguments are set up for the Earth-Moon system
    
    Assuming that the initial conditions for the target's orbit are given at apoapsis
    """
    def __init__(self,
                 period,initial_conditions_target,iter_max=15,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=0,tf=1,mu0=None,muf=None,control=False):
        
        # Convex problem parameters
        self.iter_max = iter_max
        
        # Data for the 3 body problem
        self.mu = mu
        self.LU = LU
        self.n = mean_motion
        self.TU = 1/mean_motion
        
        # Data about the target's orbit
        self.period = period
        self.initial_conditions_target = initial_conditions_target
        
        # ocp setups 
        self.n_time = n_time
        self.nx = nx
        self.nu = nu 
        self.M0 = np.radians(M0) 
        self.tf_orbit = tf   # numer of orbits
        self.control = control
        self.control_actions = None # new self value being the control actions
        if M0 < 180:
            self.ti = self.M0 * self.period / (2*np.pi) + self.period / 2
            print('less')
        if M0 > 180:
            self.ti = self.M0 *self.period /(2*np.pi) - self.period / 2
            print('more')
        if M0 == 180:
            self.ti = 0
        
        # boundary condition 
        self.μ0 = mu0 
        self.μf = muf
        
        # matrices
        self.stm = None # State Transition Matrix
        self.cim = None # Control Input Matrix
        self.psi = None # Rotation matrix synodic -> LVLH
        pass
    
    def get_traj_ref(self):
        # Generates the reference trajectory of the target spacecraft
        self.target_traj, self.time_hrz, self.dt_hrz = get_traj_ref(self.initial_conditions_target, self.M0, self.tf_orbit, self.period, self.mu, self.n_time)
        
    def linearize_trans(self):
        mats = linearize_translation(self.mu, self.target_traj, self.time_hrz, self.control)
        self.stm, self.cim, self.psi = mats["stm"], mats["cim"], mats["psi"]
        
    def get_chaser_nonlin_traj(self):
        # For now the final condition is given by propagating the non-linear dynamics in the synodic frame but conditiond are given
        # in the LVLH frame
        self.chaser_nonlin_traj = get_chaser_nonlin_traj(self.μ0, self.target_traj, self.time_hrz, self.mu)
        
    def load_traj_data(self, fname):
        # Getting the reference trajectory for the target spacecraft from a json file
        time, traj, self.mu, self.LU, self.TU = load_traj_data(fname)
        self.time_hrz = time[:self.n_time]
        self.target_traj = traj[:self.n_time]

class SCVX_OCP():
    def __init__(self,
                 period,initial_conditions_target, N_BRS, iter_max=100, koz_dim=None,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=180,tf=1,mu0=None,muf=None,control=True):
        
        # SCvx parameters, tune them to make the code run
        self.iter_max = iter_max
        self.α = np.array([2, 3])
        self.β = 2
        self.γ = 0.9
        self.ρ = np.array([0.0, 0.25, 0.7]) # 0.01 instead of 0
        self.r_minmax = np.array([1e-10, 10])
        self.εopt = 1e-4 # was at 1e-4/1e-5
        self.εfeas = 1e-4 # was at 1e-4/1e-5
        self.pen_λ = None
        self.pen_μ = None
        self.pen_w = None
        self.rk = None
        self.r0 = 1/LU # was at 2 or 0.1
        self.w0 = 100 # was at 1e4
        
        # Data for the 3 body problem
        self.mu = mu
        self.LU = LU
        self.n = mean_motion
        self.TU = 1/mean_motion
        
        # Data about the target's orbit
        self.period = period
        self.initial_conditions_target = initial_conditions_target
        
        # keep-out-zones 
        if koz_dim is not None:
            Pf = np.diag([koz_dim[0]**2, koz_dim[1]**2, koz_dim[2]**2, koz_dim[3]**2, koz_dim[4]**2, koz_dim[5]**2])
            self.inv_Pf = np.linalg.inv(Pf)
            self.N_BRS = N_BRS # Number of steps ahead we wanna ensure passive safety
            self.inv_PP = None
            self.hyperplanes = np.empty((n_time, nx, 1))
        
        # SCP setups 
        self.n_time = n_time # number of steps we're simulating for
        self.nx = nx # State vector size
        self.nu = nu # Control vector size
        self.M0 = np.radians(M0) # Initial mean motion of the target
        self.tf_orbit = tf   # numer of orbits
        self.control = control # Boolean, do we consider controls?
        
        # Computing the initial time given the initial mean motion and that data starts at apoapsis
        if M0 < 180:
            self.ti = self.M0 * self.period / (2*np.pi) + self.period / 2
            print('less')
        if M0 > 180:
            self.ti = self.M0 *self.period /(2*np.pi) - self.period / 2
            print('more')
        if M0 == 180:
            self.ti = 0
        
        # Boundary conditions for the SCP problem
        self.μ0 = mu0 
        self.μf = muf
        self.s_ref = None # Reference trajectory
        self.a_ref = None # Reference controls
        
        # Matrices
        self.stm = None # State Transition Matrix
        self.cim = None # Control Input Matrix
        self.psi = None # Rotation matrix synodic -> LVLH
        
        self.con_list = {
            "trust_region"  : False,
            "BRS"   : False
        }
        
        pass
    
    def load_traj_data(self, fname):
        # Getting the reference trajectory for the target spacecraft from a json file
        time, traj, self.mu, self.LU, self.TU = load_traj_data(fname)
        self.time_hrz = time[:self.n_time + self.N_BRS] # need more time horizon to compute the BRS for the last steps of the trajectory
        self.target_traj = traj[:self.n_time + self.N_BRS] # or n_time + N_BRS - 1?
        
    def linearize_trans(self):
        mats = linearize_translation_scvx(self.mu, self.target_traj, self.time_hrz, self.control)
        self.stm, self.cim, self.psi = mats["stm"], mats["cim"], mats["psi"]
    
    def get_unsafe_ellipsoids(self):
        self.inv_PP = np.empty((self.n_time, self.N_BRS, self.nx, self.nx))
        for i in range(len(self.s_ref)):
            ellipsoids = passive_safe_ellipsoid_scvx(self, i)
            self.inv_PP[i] = ellipsoids