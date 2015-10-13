import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

sample_num = 100

x_red, y_red = [],[]
x_blue, y_blue = [],[]

x_green, y_green = [],[]
x_yellow, y_yellow= [],[]

x, y = make_blobs(n_samples=sample_num, centers=2, n_features=2, random_state=0)

print(y[sample_num/2:])

neigh = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
neigh.fit(x[:sample_num/2],y[:sample_num/2])

#traning values
for i in range(0, sample_num/2) :
	if y[i] == 0 :
		x_red.append(x[i][0])
		y_red.append(x[i][1])
	else :
		x_blue.append(x[i][0])
		y_blue.append(x[i][1])

#predict values
for j in range(sample_num/2, sample_num) :
	if neigh.predict(x[j]) == 0 :
		x_green.append(x[j][0])
		y_green.append(x[j][1])
	else :
		x_yellow.append(x[j][0])
		y_yellow.append(x[j][1])

#draw points
plt.scatter(x_red, y_red, c='r', marker='x', s=50)
plt.scatter(x_blue, y_blue, c='b', marker='x', s=50)

plt.scatter(x_green, y_green, c='g', marker='o', s=100)
plt.scatter(x_yellow, y_yellow, c='y', marker='o', s=100)

plt.show()
