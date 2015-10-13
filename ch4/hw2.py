import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
result = np.genfromtxt("data.txt", usecols=(13))

data = preprocessing.scale(data)
print data

for i in range (1,7) :

	polynomial_features = PolynomialFeatures(degree=i)
	linear_regression = LinearRegression(fit_intercept=False)

	pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
	pipeline.fit(data, result)

	print ("r2_score : %f" % r2_score(result, pipeline.predict(data)))

#print pipeline

#print pipeline.named_steps['linear_regression'].coef_
#print len(pipeline.named_steps['linear_regression'].coef_)
#print pipeline.named_steps['linear_regression'].intercept_

#print (r2_score(result, pipeline.predict(data)))
