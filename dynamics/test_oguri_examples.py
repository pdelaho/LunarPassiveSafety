import numpy as np

from setup_oguri_examples import *

prob = Example1(w0=100)

sol_0 = {"z": np.asarray([1.5, 1.5]), "l": 0, "status": True, "f0": 3, "P": 0, "value": 0}
zref = np.asarray([1.5, 1.5])

prob, log, k = scvx_star(prob, sol_0, zref)

# not working because never converges, fix it
