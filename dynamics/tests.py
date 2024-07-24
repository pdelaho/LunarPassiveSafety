import numpy as np

B = np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)
print(B,B.shape)