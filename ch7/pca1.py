import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as la

X = np.array([[0.9,1], [2.4, 2.6], [1.2, 1.7], [0.5, 0.7], [0.3, 0.7], [1.8, 1.4], [0.5, 0.6], [0.3, 0.6], [2.5, 2.6], [1.3, 1.1]])

mean_x = np.mean(X, axis=0)


for i in range(0, len(X)):
	X[i][0] = X[i][0]-mean_x[0]
	X[i][1] = X[i][1]-mean_x[1]

cov = np.cov(X.T)
#print cov

w,v = la.eig(cov)

v = v.T
print v[0]

dot = np.dot(v[0], X.T)

print dot

pca = PCA(n_components=2)
result = pca.fit(X)

print result

print(pca.components_)
print(pca.explained_variance_ratio_)
