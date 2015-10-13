import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import preprocessing

data = np.genfromtxt("data.txt", usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
result = np.genfromtxt("data.txt", usecols=(13))

#data = preprocessing.scale(data)

print data

alpha=[0.025,0.03,0.035,0.036,0.037,0.038,0.039,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.7,0.8,0.9,1.0,5.0,6.0,7.0,8.0,9.0,10.0]

clf = linear_model.RidgeCV(alpha)
clf.fit(data, result)

clf1 = linear_model.Ridge(clf.alpha_)
#clf1 = linear_model.Ridge(0.5)
clf1.fit(data, result)

print ("alpha : %f" %clf.alpha_)

for i in range(0, 13) :
	print clf1.coef_[i]
print ("intercept : %f " % clf1.intercept_)

print ("r2_score : %f" %r2_score(result, clf1.predict(data)))
