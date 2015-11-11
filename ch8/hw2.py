import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

threshold = 1

model = LinearRegression()
model.fit(data, result)

#for i in range(0,8):
#	print model.coef_[i]

#print ("intercept : %f"%model.intercept_)

model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression())
print model_ransac
model_ransac.fit(data,result)
inliner_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inliner_mask)

print len(data)
true_count=0
false_count=0

for i in range(0, len(data)):
	if inliner_mask[i] :
		true_count += 1
	else : 
		false_count += 1

print("true_count : %d"%true_count)
print("false_count : %d"%false_count)
#print model_ransac.estimator_.coef_

residual = result - model.predict(data)
#print residual
