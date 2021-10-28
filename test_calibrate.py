import numpy as np

a = np.mgrid[0:260:14j, 0:120:7j,  0:300:15j]
b = a.T.reshape(-1,7,3)
print(b)
