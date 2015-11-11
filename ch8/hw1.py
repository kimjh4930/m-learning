import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

model = LinearRegression()
model.fit(data, result)

#print(model.predict(data))

model_ransac=linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), residual_threshold=20)
model_ransac.fit(data, result)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(-5,5)
line_y = model.predict(line_X)


#plt.figure()
#plt.plot(data, result, '-g')
#plt.plot(data, model.predict(data), '-k', label='linear regressor')
#plt.plot(data, model_ransac.predict(data), '-r', label='RANSAC regressor')

#plt.legend(loc='upper left')
#plt.show()

#print data
#print result
