from sklearn import preprocessing

import numpy as np

x = np.array([[1., -1., 2.],
							[2., 0., 0.],
							[0., 1., -1.]])

x_scaled = preprocessing.scale(x)

print(x_scaled)
