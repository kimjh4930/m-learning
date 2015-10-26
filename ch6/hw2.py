import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from itertools import cycle

data = np.genfromtxt("data.txt",usecols=(0,1,2,3,4,5,6))
result = np.genfromtxt("data.txt", usecols=7)

K=range(1,20)
meanDistortions = []

k_means=KMeans(n_clusters=3)
k_means.fit(data)

label = k_means.labels_[:]
labels_unique = k_means.labels_
n_clusters_ = len(labels_unique)
cluster_centers = k_means.cluster_centers_

for i in range(0,len(label)):
	label[i] = label[i] + 1

print(k_means.labels_[:])

accuracy = accuracy_score(result, label[:])

print('accuracy : %0.2f'%(accuracy))

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
	my_members = label == k
	cluster_center = cluster_centers[k]
	plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
	plt.plot(cluster_center[0], cluster_center[1], 'o',markerfacecolor=col, markeredgecolor='k', markersize=14)
