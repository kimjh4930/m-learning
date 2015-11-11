import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

model = LinearRegression()
print model
model.fit(data, result)

for i in range(0,8):
	print model.coef_[i]

print ("intercept : %f"%model.intercept_)

model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), residual_threshold=100)
model_ransac.fit(data,result)

for i in range(0,8):
	print model_ransac.coef_[i]

print ("intercept : %f"%model.intercept_)


