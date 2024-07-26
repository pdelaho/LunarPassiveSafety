import numpy as np 
import sys
import os 

from useful_small_functions import *

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

fname = root_folder + "/dynamics/data/traj.json"

t, state, mu, LU, TU = load_traj_data(fname)
print("traj loaded...")
print(f"mu: {mu}")   