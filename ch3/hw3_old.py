import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing

data = np.genfromtxt("data.txt",delimiter=",",usecols=(0,1,2,3,4))
result = np.genfromtxt("data.txt", delimiter=",", usecols=5)

#for i in range(18,30) : 
train_num = 120

#standardization

#data = preprocessing.scale(data)

#KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[:train_num], result[:train_num])

accuracy = accuracy_score(result[train_num:], neigh.predict(data[train_num:]))

print("KNN accuracy : %0.2f " %(accuracy))

#linear svm
#set kernel
clf = svm.SVC(kernel='linear', C=1).fit(data[:train_num], result[:train_num])

accuracy = accuracy_score(result[train_num:], clf.predict(data[train_num:]))

print('Linear svm accuracy : %0.2f' %(accuracy))

#k-fold cross-validation

score = cross_validation.cross_val_score(clf, data[train_num:], result[train_num:], cv=5)
print(score)
print('Cross validation : %0.2f' %score.mean())
