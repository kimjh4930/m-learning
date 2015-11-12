import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets
from sklearn.metrics import r2_score

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7))
result = np.genfromtxt("data.txt", usecols=(8))

result_sum = 0
mean=0
result_=0
result_x = list()
result_y = list()
ransac_true_index = list()
ransac_true_list = list()
ransac_true_result = list()
result1=list()
result2=list()

for i in range(0, len(result)) :
	result_sum += result[i]
	
mean = result_sum/len(result)
 
for j in range(0, len(result)) :
	result_ = result_ + abs(result[j]-mean)

MAD = result_/len(result)

trials = list()
trials.append(2*MAD)
trials.append(3*MAD)

print(MAD)
for j in range(0,200):
	for i in range(0,1):

		model = LinearRegression()
		model.fit(data, result)

		model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
																						max_trials = 200,residual_threshold=2*MAD)
	
		model_ransac.fit(data,result)
		inliner_mask = model_ransac.inlier_mask_
		outlier_mask = np.logical_not(inliner_mask)

	#print("Linear Regression r2_score : %f"%r2_score(result, model.predict(data)))

		true_count=0
		false_count=0
		ransac_true_index[:]=[]
		for k in range(0, len(data)):
			if inliner_mask[k] :
				true_count += 1
				ransac_true_index.append(k)
			else : 
				false_count += 1
	
		ransac_true_list[:]=[]
		ransac_true_result[:]=[]
		for l in range(0, len(ransac_true_index)):
			ransac_true_list.append(data[ransac_true_index[l]])
			ransac_true_result.append(result[ransac_true_index[l]])
		
		r2 = r2_score(ransac_true_result, model_ransac.predict(ransac_true_list))
		
		if i==0 :
			result1.append(true_count)
			result2.append(r2)
		else :
			result3.append(true_count)
			result4.append(r2)

result_sum = 0.0
r2_sum = 0.0
result_av = 0.0
r2_av=0.0

for m in range(0, len(result1)):
	result_sum += result1[m]
	r2_sum += result2[m]

result_av = result_sum/len(result1)
result_r2 = r2_sum/len(result1)

print("Linear Regression r2_score : %f"%r2_score(result, model.predict(data)))
print("RANSAC LinearRegression r2_score : %f"%result_r2)
print("average num of true : %f"%result_av)

