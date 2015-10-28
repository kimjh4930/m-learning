import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as la

X = np.array([[0.9,1], [2.4, 2.6], [1.2, 1.7], [0.5, 0.7], [0.3, 0.7], [1.8, 1.4], [0.5, 0.6], [0.3, 0.6], [2.5, 2.6], [1.3, 1.1]])

pca =PCA(n_components=1)
result = pca.fit_transform(X)

print result
