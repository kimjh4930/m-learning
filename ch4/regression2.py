# polynomial regression : y = -8.39 + 2.95x+0.0082x^2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

x=[[6,1],[8,2],[10,1],[14,4],[18,5]]
y=[[7],[9],[13],[17.5],[18]]

model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
#model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model.fit(x,y)

print model.named_steps['linear'].coef_
print len(model.named_steps['linear'].coef_)
#print r2_score(y, model.predict(x))

#model_1=LinearRegression()
#model_1.fit(x,y)
#print model_1.coef_
#print model_1.intercept_

print(r2_score(y,model.predict(x)))

plt.figure()
plt.plot(x, y, 'k.')
plt.plot(x, model.predict(x), color='blue', linewidth=3)
plt.show()
