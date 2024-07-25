import dataclasses

from dynamics_linearized import *
from get_initial_conditions import *


class CR3BP_RPOD_OCP:
    """
    Optimal Control Problem (OCP) in the context of 3-body problem class
    Default arguments are set up for the Earth-Moon system
    
    Assuming that the initial conditions for the target's orbit are given at apoapsis
    """
    def __init__(self,
                 period,initial_conditions_target,iter_max=15,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,M0=0,tf=1,mu0=None,muf=None):
        
        # SCP parameters
        self.iter_max = iter_max
        
        # Data for the 3 body problem
        self.mu = mu
        self.LU = LU
        self.n = mean_motion
        self.TU = 1/mean_motion
        
        # Data abou the target's orbit
        self.period = period
        self.initial_conditions_target = initial_conditions_target
        
        # ocp setups 
        self.n_time = n_time
        self.nx = nx
        self.nu = nu 
        self.M0 = M0 
        self.tf_orbit = tf   # numer of orbits
        
        # boundary condition 
        self.μ0 = mu0 
        self.μf = muf
        
        # matrices
        self.stm = None # State Transition Matrix
        self.cim = None # Control Input Matrix
        self.psi = None # Rotation matrix synodic -> LVLH
        pass
    
    def get_traj_ref(self, n_time):
        # Generates the reference trajectory of the target spacecraft
        self.target_traj, self.time_hrz, self.dt_hrz = get_traj_ref(self.initial_conditions_target, self.M0, self.tf_orbit, self.period, self.mu, n_time)
        
    def linearize_trans(self):
        mats = linearize_translation(self.mu, self.target_traj, self.time_hrz)
        self.stm, self.cim, self.psi = mats["stm"], mats["cim"], mats["psi"]
        
    def get_final_condition(self):
        # For now the final condition is given by propagating the non-linear dynamics in the synodic frame but conditiond are given
        # in the LVLH frame
        self.μf = get_final_condition(self.μ0, self.target_traj, self.time_hrz, self.mu)