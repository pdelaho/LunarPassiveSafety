import dataclasses

from dynamics_linearized import *
# from get_initial_conditions import *


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
        self.M0 = np.radians(M0) 
        self.tf_orbit = tf   # numer of orbits
        self.control = control
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
        # ti_idx = np.argmin([abs(i-self.ti) for i in time]) # finding the index with the closest time step to our desired initial time
        # self.ti_idx = ti_idx
        # if len(time) - ti_idx - 1 - self.n_time > 0:
        #     self.time_hrz = time[ti_idx : ti_idx + self.n_time]
        #     self.target_traj = traj[ti_idx : ti_idx + self.n_time]
        # else:
        #     self.time_hrz = time[ti_idx:]
        #     self.target_traj = traj[ti_idx:]
        # print(ti_idx, self.time_hrz, self.target_traj[0])
        self.time_hrz = time[:self.n_time]
        self.target_traj = traj[:self.n_time]
