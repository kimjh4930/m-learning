import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm

sample_num = 50;

x_red, y_red = [],[]
x_blue, y_blue = [],[]

x, y = make_blobs(n_samples=sample_num, centers=2, n_features=2, random_state=2)

clf = svm.SVC(kernel='linear')
print clf
clf.fit(x, y)

#result of lagrange multipliers, lagrange value : alpha
print clf.dual_coef_

#support vector
print clf.support_vectors_

#coef, intercept_
print clf.coef_
print clf.intercept_

for j in range(0, len(y)):
	if y[j] == 0 :
		y[j] = -1

for i in range(0, sample_num) :
	if y[i] == -1 :
		x_red.append(x[i][0])
		y_red.append(x[i][1])
	else : 
		x_blue.append(x[i][0])
		y_blue.append(x[i][1])

plt.scatter(x_red, y_red, c='r', marker='o', s=50)
plt.scatter(x_blue, y_blue, c='b', marker='o', s=50)

x = np.arange(10)
y = clf.coef_*x + clf.intercept_

plt.plot(x,y,'-')

plt.show()
