import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

#generate sample data
centers = [[1,1],[-1,-1],[1,-1]]
#X,_ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
X,Y = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

print Y
#Compute clustering with MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)

print ("bandwidth : %f"% bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
print(labels)
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimate cluster : %d"% n_clusters_)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
	my_members = labels == k
	cluster_center = cluster_centers[k]
	plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
	plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

plt.title('Estimated number of clusters: %d' %n_clusters_)
plt.show()
