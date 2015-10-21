import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

data = np.genfromtxt("data.txt",usecols=(0,1,2,3,4,5,6))
result = np.genfromtxt("data.txt", usecols=7)

print data
print result

K=range(1,20)
meanDistortions = []

for k in K:
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(data)
	SSE = sum(np.min(cdist(data, kmeans.cluster_centers_,'euclidean'), axis=1)) / data.shape[0]
	meanDistortions.append(SSE) #value of SSE
	print SSE

print kmeans.fit(data)

plt.plot(K, meanDistortions, 'bx-')
plt.show()
