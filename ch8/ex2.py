import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

n_samples = 1000
n_outliers = 50

x, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)

np.random.seed(0)

x[:n_outliers] = 3 + 0.5*np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10*np.random.normal(size=n_outliers)

model = linear_model.LinearRegression()
model.fit(x,y)
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(x,y)

inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(-5, 5)
line_y = model.predict(line_X[:,np.newaxis])
line_y_ransac = model_ransac.predict(line_x[:, np.newaxis])

print("Estimated coefficients (true, normal, RANSAC) : ")
print(coef, model.coef_, model_ransac.estimator_.coef_)

plt.plot(X[])
