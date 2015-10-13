import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
result = np.genfromtxt("data.txt", usecols=(13))

#data = preprocessing.scale(data)

model = LinearRegression()
model.fit(data, result)

print data

for i in range(0,13) :
	print (model.coef_[i])


print ("intercept : %f" % model.intercept_)

print ("r2_score : %f" % r2_score(result, model.predict(data)))

