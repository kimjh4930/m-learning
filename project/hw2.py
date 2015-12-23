import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

sample_num = 200
cluster_std = 2
random_state = 1

dist_lp = []

x, y = make_blobs(n_samples=sample_num, centers=2, n_features=2, random_state=random_state, cluster_std=cluster_std)

x[50] = [-7.0, 0.0]
print y[50]

clf = svm.SVC(kernel='linear')
clf.fit(x, y)


w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-12,5)
yy = a * xx - (clf.intercept_[0])/w[1]

print a
print clf.intercept_[0]

# need not to modifying
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy -a * margin

plt.clf()
plt.plot(xx, yy, 'k-')
#plt.plot(xx1, yy1, 'r--')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)

plt.scatter(x[:,0], x[:,1], c=y, zorder=10, cmap=plt.cm.Paired)
#plt.scatter(x_outlier[:,0], x_outlier[:,1], c=y, zorder=10, cmap=plt.cm.Paired)

plt.show()
