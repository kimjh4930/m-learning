import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la
from sys import maxint

sample_num = 100
x, y = make_blobs(n_samples=sample_num, centers=3, n_features=3, random_state=0)

mean = np.mean(x, axis=0)

#print "np.mean(x, axis=0)"
#print mean


x = x - mean

cov = np.cov(x.T)

w,v = la.eig(cov)

big_1 = np.argmax(w)
w[big_1] = -maxint-2
big_2 = np.argmax(w)

eig_matrix = np.array([v[big_1], v[big_2]])
print eig_matrix

v = v.T
print v

result = np.array([np.dot(v[big_1], x.T), np.dot(v[big_2], x.T)]).T

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

