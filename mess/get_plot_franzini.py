import setup_path   ## IMPORTANT: make sure to access /src/
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from test_stm_nonlin import *
from dynamics.get_initial_conditions import *

LU = 384400 # km, distance between primary attractors
mu = 1.215e-2 # no unit, mass parameter of the system
TU = 1/(2.661699e-6) # s, inverse of the relative angular frequency between the two primary attractors

M = np.linspace(0,360,5)
M_labels = [f"{int(mean_anomaly)}" for mean_anomaly in M ]
print(M_labels)
base = np.asarray([i for i in range(10,1,-1)])
# rho_init = np.concatenate(([1e-2],1e-2*base, 1e-1*base, base, 10*base))
# rho_init = np.concatenate((10*base, base, 1e-1*base, 1e-2*base, [1e-2]))
rho_init = np.asarray([10,5,1,0.5,0.1])
rho_init_labels = [f"{rho} [km]" for rho in rho_init]
print(rho_init_labels)

mat_dist = np.empty((len(M),len(rho_init))) # just store the maximum error on the 12 hour propagation
mat_vel = np.empty((len(M),len(rho_init)))


for i in range(len(M)):
    print(i)
    for j in range(len(rho_init)):
        print(j)
        errors_dist = np.empty(5)
        errors_vel = np.empty(5)
        for k in range(5):
            IC_target_M, IC_chaser_LVLH, IC_chaser_M = get_initial_conditions(np.radians(M[i]),rho_init[j],0)
            error_distance, error_velocity = verification(IC_target_M, IC_chaser_LVLH, IC_chaser_M)
            errors_dist[k] = max(error_distance)
            errors_vel[k] = max(error_velocity)
        mat_dist[i,j] = math.floor(math.log10(np.average(errors_dist)*LU*1e3)) # see to use math.floor in addition
        mat_vel[i,j] = math.floor(math.log10(np.average(errors_vel)*LU*1e3/TU))

ax2 = sns.heatmap(mat_dist.T, linewidth=0.5)

fig, ax = plt.subplots()
im = ax.imshow(mat_dist.T)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$log_{10}(||\hat{\rho} - \rho||)$", rotation=-90, va="bottom")
ax.set_xticks(np.arange(len(M_labels)), labels=M_labels)
ax.set_yticks(np.arange(len(rho_init_labels)), labels=rho_init_labels)
fig.tight_layout
plt.title(r"Comparing the relative position from linear and non-linear dynamics")
plt.show()

ax2 = sns.heatmap(mat_vel.T, linewidth=0.5)

fig, ax = plt.subplots()
im = ax.imshow(mat_vel.T)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$log_{10}(||\hat{\dot{\rho}} - \dot{\rho}||)$", rotation=-90, va="bottom")
ax.set_xticks(np.arange(len(M_labels)), labels=M_labels)
ax.set_yticks(np.arange(len(rho_init_labels)), labels=rho_init_labels)
fig.tight_layout
plt.title(r"Comparing the relative velocity from linear and non-linear dynamics")
plt.show()