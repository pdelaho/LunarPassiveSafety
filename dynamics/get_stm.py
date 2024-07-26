import numpy as np 
import sys
import os 

from useful_small_functions import *
from dynamics_linearized import *
from ocp import * 


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

fname = root_folder + "/dynamics/data/9_2_S_Halo.json"

t, traj, mu, LU, TU = load_traj_data(fname)
print("traj loaded...")
print(f"mu: {mu}")   
print(f"LU: {LU}")
print(f"TU: {TU}")  
print(np.shape(traj))

mats = linearize_translation(mu, traj, t, True)

class OCP:
    def __init__(self, mats):
        self.stm = mats["stm"]
        self.cim = mats["cim"]
        self.psi = mats["psi"]  # LVLH -> Synodic rotation matrix
        self.μ0 = None
        self.μf = None
        self.n_time = len(t)
        self.nu = 3
        self.nx = 6 


prob = OCP(mats)
sol = ocp_cvx(prob)

mu = sol["mu"]



print("done!!")



