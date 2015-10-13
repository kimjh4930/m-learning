import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

sample_num = 100

x_red, y_red = [],[]
x_blue, y_blue = [],[]

x, y = make_blobs(n_samples=sample_num, centers=2, n_features=2, random_state=0)

print(y[sample_num/2:])

for i in range(0, sample_num) :
	if y[i] == 0 :
		x_red.append(x[i][0])
		y_red.append(x[i][1])
	else :
		x_blue.append(x[i][0])
		y_blue.append(x[i][1])

#draw points

plt.scatter(x_red, y_red, c='r', marker='x', s=50)
plt.scatter(x_blue, y_blue, c='b', marker='x', s=50)

plt.show()
