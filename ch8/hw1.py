import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

model = LinearRegression()
model.fit(data, result)

print len(data)

print "LinearRegression"
print("coef")
for i in range(0,8):
	print model.coef_[i]
print("intercept : %f"%model.intercept_)

model_ransac=linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression())
print model_ransac
model_ransac.fit(data, result)

print " "
print "RANSAC LinearRegression"
print "coef"
for i in range(0,8):
	print model_ransac.estimator_.coef_[i]
print("intercept : %f"%model_ransac.estimator_.intercept_)


