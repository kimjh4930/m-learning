import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

result_sum = 0
mean=0
result_=0
result_x = list()
result_y = list()

for i in range(0, len(result)) :
	result_sum += result[i]
	
mean = result_sum/len(result)
 
for j in range(0, len(result)) :
	result_ = result_ + abs(result[j]-mean)

MAD = result_/len(result)

print(MAD)

for index in range(1, 10):
	threshold = index*20

	model = LinearRegression()
	model.fit(data, result)

	model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),max_trials = threshold,residual_threshold = MAD)
#	print model_ransac
	
	model_ransac.fit(data,result)
	inliner_mask = model_ransac.inlier_mask_
	outlier_mask = np.logical_not(inliner_mask)

	print("max_trials : %d"%threshold)
	print "coef"
	for i in range(0,8):
		print model_ransac.estimator_.coef_[i]
	print("intercept : %f"%model_ransac.estimator_.intercept_)

	true_count=0
	false_count=0

	for i in range(0, len(data)):
		if inliner_mask[i] :
			true_count += 1
		else : 
			false_count += 1
	result_x.append(index)
	result_y.append(true_count)

plt.plot(result_x, result_y, 'bo')
plt.show()
