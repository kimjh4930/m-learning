import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la
from sklearn.decomposition import PCA

sample_num = 100
x, y = make_blobs(n_samples=sample_num, centers=3, n_features=3, random_state=0)

mean = np.mean(x, axis=0)

pca = PCA(n_components=2)
result = pca.fit_transform(x)
print pca.components_

#print pca.explained_variance_ratio_
#print result

red_x, red_y = [],[]
blue_x, blue_y = [],[]
green_x, green_y = [],[]

#print len(result)
#print len(y)

for i in range(len(result)) :
	if y[i] == 0:
		red_x.append(result[i][0])
		red_y.append(result[i][1])
	elif y[i] == 1:
		green_x.append(result[i][0])
		green_y.append(result[i][1])
	else :
		blue_x.append(result[i][0])
		blue_y.append(result[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='o')
plt.scatter(green_x, green_y, c='g', marker='.')

plt.show()

