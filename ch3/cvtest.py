import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)


#print(iris.data)

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

#print(clf)

#print(len(x_train),len(x_test),len(iris.data), len(iris.target))
#print(clf.score(x_test, y_test))

print(scores)
