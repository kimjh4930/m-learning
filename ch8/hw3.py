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

print("MAD : %f"%MAD)

for index in range(1, 100):
	threshold = index

	model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
																							residual_threshold=3*MAD)
	model_ransac.fit(data,result)
	inliner_mask = model_ransac.inlier_mask_
	outlier_mask = np.logical_not(inliner_mask)

	true_count=0
	false_count=0

	for i in range(0, len(data)):
		if inliner_mask[i] :
			true_count += 1
		else : 
			false_count += 1
	if threshold < MAD :
		print("threshold : %d "%threshold),
		print("true_count : %d "%true_count),
		print("false_count : %d"%false_count)
	result_x.append(index)
	result_y.append(true_count)

plt.plot(result_x, result_y, 'bo')
plt.show()

#print model_ransac.estimator_.coef_

#residual = result - model.predict(data)
#print residual

