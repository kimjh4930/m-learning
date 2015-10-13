from sklearn import linear_model
import pylab as pl

x=[[0,0], [0,0], [0,1]]
y=[[0],[0.1],[1]]

clf = linear_model.Ridge(alpha=0.5)
clf.fit(x, y)

print clf.coef_
print clf.intercept_
