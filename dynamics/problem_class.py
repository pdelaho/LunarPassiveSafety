from linear_dynamics_LVLH import *

class CR3BP_RPOD_OCP:
    """
    Optimal Control Problem (OCP) in the context of 3-body problem class
    Default arguments are set up for the Earth-Moon system
    """
    def __init__(self,
                 period,initial_conditions_target,iter_max=15,
                 mu=1.215e-2,LU=384400,mean_motion=2.661699e-6,
                 n_time=100,nx=6,nu=3,t0=0,tf=1,mu0=None,muf=None):
        
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
        self.t0 = t0 
        self.tf_orbit = tf   # numer of orbits
        
        # boundary condition 
        self.μ0 = mu0 
        self.μf = muf
        
        # matrices
        self.stm = None # State Transition Matrix
        self.cim = None # Control Input Matrix
        self.rot = None # Rotation matrix synodic -> LVLH
        pass