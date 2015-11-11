import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

x=[[18], [6], [30], [6], [36], [18], [18], [30], [36], [50], [55]]
y=[[25], [17], [48], [23], [40], [30], [39], [40], [60], [10], [15]]

model=LinearRegression()
model.fit(x, y)
model_ransac=linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(x, y)

plt.figure()
plt.plot(x, y, '.g')
plt.plot(x, model.predict(x), '-k', label='linear regressor')
plt.plot(x, model_ransac.predict(x), '-r', label='RANSAC regressor')

plt.legend(loc='upper left')
plt.show()
