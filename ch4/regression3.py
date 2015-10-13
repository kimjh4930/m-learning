import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(0)

n_samples = 30
degrees = [1,4,15]

true_fun = lambda X: np.cos(1.5 * np.pi * X)
x = np.sort(np.random.rand(n_samples))
y = true_fun(x) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 4))

for i in range(len(degrees)):
	ax = plt.subplot(1, len(degrees), i+1)
	plt.setp(ax, xticks=(), yticks=())
	polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)

	linear_regression = LinearRegression()
	pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
	pipeline.fit(x[:,np.newaxis], y)

	x_test = np.linspace(0,1,100)

	plt.plot(x_test, pipeline.predict(x_test[:,np.newaxis]), label="Model")
	print x_test
	print x_test[:, np.newaxis]
	plt.plot(x_test, true_fun(x_test), label="True function")
	plt.scatter(x, y, label="samples")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim((0,1))
	plt.ylim((-2,2))
	#plt.legend(loc="best")
	#plt.title("Degree %d" % degrees[i])
plt.show()
