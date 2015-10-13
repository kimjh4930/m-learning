import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from numpy.linalg import *

x, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

print(np.cov(np.array(x).T))

print(inv(np.cov(np.array(x).T)))
