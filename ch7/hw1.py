import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la

sample_num = 100
x, y = make_blobs(n_samples=sample_num, centers=3, n_features=3, random_state=0)

mean = np.mean(x, axis=0)

for i in range(0, 100):
	x[i][0] = x[i][0]-mean[0]
	x[i][1] = x[i][1]-mean[1]
	x[i][2] = x[i][2]-mean[2]

cov = np.cov(x.T)
#print (cov)

w,v = la.eig(cov)

print w
v = v.T
print v[0]
print v[2]

product1 = np.dot(v[0], x.T)
product2 = np.dot(v[2], x.T)

result = np.array([np.dot(v[0], x.T), np.dot(v[2], x.T)]).T

print result
#'w' is eigenvalue
#'v' is eigenvector

#print v

red_x, red_y = [],[]
blue_x, blue_y = [],[]
green_x, green_y = [],[]

for i in range(len(result)) :
	if y[i] == 0:
		red_x.append(result[i][0])
		red_y.append(result[i][1])
	elif y[i] == 1:
		green_x.append(result[i][0])
		green_y.append(result[i][1])
	else :
		blue_x.append(result[i][0])
		blue_y.append(result[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='o')
plt.scatter(green_x, green_y, c='g', marker='.')

plt.show()

