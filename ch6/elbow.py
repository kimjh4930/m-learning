import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

cluster1=np.random.uniform(0.5, 100,(2,50))
#cluster2=np.random.uniform(3.5, 150,(2,50))

print cluster1

#x=np.hstack((cluster1, cluster2)).T
#print x
#x=np.vstack((cluster1, cluster2)).T
#print x



K=range(1,20)
meandistortions = []

for k in K:
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(x)
	meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_,'euclidean'),axis=1)) / x.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.show()
