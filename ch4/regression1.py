import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x=[[6,2],[8,1],[10,0],[14,2],[18,0]]
y=[[7],[9],[13],[17.5],[18]]

model=LinearRegression()
model.fit(x,y)

print (model.coef_)
print (model.intercept_)

model.predict([8,2])
print(r2_score(y, model.predict(x)))

plt.figure()
plt.plot(x, y, 'k.')
plt.plot(x, model.predict(x), color='blue', linewidth=3)
plt.show()
