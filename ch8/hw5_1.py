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

for i in range(0, len(result)) :
	result_sum += result[i]
	
mean = result_sum/len(result)
 
for j in range(0, len(result)) :
	result_ = result_ + abs(result[j]-mean)

MAD = result_/len(result)

print(MAD)

model = LinearRegression()
model.fit(data, result)

print("LinearRegression r2_score : %f" %r2_score(result, model.predict(data)))

model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),max_trials = 200, residual_threshold=MAD)
	
model_ransac.fit(data,result)
inliner_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inliner_mask)

print ("r2_score : %f"%r2_score(ransac_true_result, model_ransac.predict(ransac_true_list)))

