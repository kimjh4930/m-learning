import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

data = np.genfromtxt("data.txt",usecols=(0,1,2,3,4,5,6))
result = np.genfromtxt("data.txt", usecols=7)

K=range(1,20)
meanDistortions = []

k_means=KMeans(n_clusters=3)
k_means.fit(data)

label = k_means.labels_[:]

for i in range(0,len(label)):
	label[i] = label[i] + 1

print(k_means.labels_[:])

accuracy = accuracy_score(result, label[:])

print('accuracy : %0.2f'%(accuracy))



