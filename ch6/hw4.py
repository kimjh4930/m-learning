import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.metrics import accuracy_score

data = np.genfromtxt("data.txt",usecols=(0,1,2,3,4,5,6))
result = np.genfromtxt("data.txt", usecols=7)

bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=len(data))

print ("bandwidth : %f"% bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
print ms.fit(data)

labels = ms.fit_predict(data)

for i in range(0, len(labels)):
	if labels[i] == 0 :
		labels[i] = 3
	elif labels[i] == 1 :
		labels[i] = 1
	else:
		labels[i] = 2

print ("labels : ",labels)

cluster_centers = ms.cluster_centers_
print ("cluster_centers : ", cluster_centers)

labels_unique = np.unique(labels)

print("labels_unique : ", labels_unique)
n_clusters_ = len(labels_unique)

print("number of estimate cluster : %d"% n_clusters_)

#accuracy
accuracy = accuracy_score(result, labels)
print("accuracy : %f"% accuracy)



