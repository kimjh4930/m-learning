from sklearn import linear_model

#clf = linear_model.RidgeCV(alphas=[0.1,1.0,10.0])
clf = linear_model.RidgeCV(alphas=[1,2,3,4,5,6,7,8,9])
clf.fit([[0,0],[0,0],[1,1]],[0,.1,1])
#print RidgeCV(alphas=[.0,1.0,10.0], cv=None, fit_intercept=True, scoring=None, normalize=False)
print clf
print(clf.alpha_)
