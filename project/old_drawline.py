import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
#X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
#Y = [0] * 20 + [1] * 20

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# figure number
fignum = 1

#X[98] = [-5.0,0]
#Y[98] = 0

# fit the model
#for name, penalty in (('unreg', 1), ('reg', 0.05)):
#for i in (0, 0):
	#clf = svm.SVC(kernel='linear', C=penalty)
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, Y)

#print(clf.fit(X,Y))

# get the separating hyperplane
w = clf.coef_[0]
print('----- clf.coef in drawline.py ----')
print (w)
a = -w[0] / w[1]
print('----- -w[0]/w[1] in drawline.py ----')
print ("a : %f"%a)
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

#lagrange multipliers, alpha
	#print clf.dual_coef_

# plot the parallels to the separating hyperplane that pass through the
# support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

# plot the line, the points, and the nearest vectors to the plane
	#plt.figure(fignum, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, 'k-')
#plt.plot(xx, yy_down, 'k--')
#plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
												facecolors='none', zorder=10)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, zorder=10, cmap=plt.cm.Paired)
'''
	plt.axis('tight')
	x_min = -4.8
	x_max = 4.2
	y_min = -6
	y_max = 6

	XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
	Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
	Z = Z.reshape(XX.shape)
	plt.figure(fignum, figsize=(4, 3))
	plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	plt.xticks(())
	plt.yticks(())
	fignum = fignum + 1
'''
plt.show()
