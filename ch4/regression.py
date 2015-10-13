import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x=[[6],[8],[10],[14],[18]]
y=[[7],[9],[13],[17.5],[18]]

model = LinearRegression()
model.fit(x,y)

print model.coef_
print model.intercept_

plt.figure()
plt.plot(x, y, 'k.')
print(r2_score(y, model.predict(x)))
plt.plot(x, model.predict(x), color='blue', linewidth=3)
plt.show()
