from sklearn import svm

x = [[0,0], [1,1]]
y = [0,1]

clf = svm.LinearSVC()
clf.fit(x,y)
print(clf.predict([[2., 2.,]]))
print(clf.predict([[-1.,-1.]]))
