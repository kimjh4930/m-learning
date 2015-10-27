import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

sample_num = 100
x, y = make_blobs(n_samples=sample_num, centers=3, n_features=3, random_state=0)

print (np.cov(x.T))

mean = np.mean(x, axis=0)

for i in range(0, len(mean)):
	x[i][0] = x[i][0]-mean[0]
	x[i][1] = x[i][1]-mean[1]
	x[i][2] = x[i][2]-mean[2]

print x

cov = np.cov(x)
print cov
