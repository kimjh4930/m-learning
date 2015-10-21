import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

k_means=cluster.KMeans(n_clusters=3)
k_means.fit(x)

print(k_means.labels_[:])
print(y)
