import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()

x=iris.data
y=iris.target

pca = PCA(n_components=2)

reduced_x = pca.fit_transform(x)
red_x, red_y = [],[]
blue_x, blue_y = [],[]
green_x, green_y = [],[]

for i in range(len(reduced_x)):
	if y[i] == 0:
		red_x.append(reduced_x[i][0])
		red_y.append(reduced_x[i][1])
	elif y[i] == 1:
		blue_x.append(reduced_x[i][0])
		blue_y.append(reduced_x[i][1])
	else : 
		green_x.append(reduced_x[i][0])
		green_y.append(reduced_x[i][1])

plt.scatter(blue_x, blue_y, c='b', marker='o')
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(green_x, green_y, c='g', marker='.')

plt.show()
