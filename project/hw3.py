import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

sample_num = 190
cluster_std = 2
random_state = 1

dist_lp = []

x1, y1 = make_blobs(n_samples=sample_num, centers=2, n_features=2, random_state=random_state, cluster_std=cluster_std)

x_outlier = np.array([[-13.106,7.130],
											[-8.521,7.576],
											[-5.662,0.688],
											[-0.0645,-0.778],
											[6.963,-6.773],
											[-9.176,-3.840],
											[-5.603,-1.289],
											[-4.531,-3.202],
											[-4.531,-3.202],
											[-10.1,-1.07]])
y_outlier = np.array([1,1,1,1,1,0,0,0,0,0])

x_outlier = x_outlier.tolist()
y_outlier = y_outlier.tolist()

x = x1.tolist()
y = y1.tolist()

for i in range(0, len(x_outlier)) :
	x.append(x_outlier[i])
	y.append(y_outlier[i])

#x[50]=[-6.77, 3.59184]
#print y[50]

for i in range(45,55):
	print y[i]

clf = svm.SVC(kernel='linear')
clf.fit(x, y)

#distance function
def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-12,5)
yy = a * xx - (clf.intercept_[0])/w[1]

print "before remove outlier"
print a
print -(clf.intercept_[0])/w[1]

#distance point of A to mean of B
temp_g0 = []
temp_g1 = []
index_g0 = []
index_g1 = []

for i in range(0, len(x)):
	if y[i] == 0 :
		temp_g0.append(x[i])
		index_g0.append(i)
	else :
		temp_g1.append(x[i])
		index_g1.append(i)

mean_g0 = np.mean(temp_g0, axis=0)
mean_g1 = np.mean(temp_g1, axis=0)
distance_g0p1 = []
distance_g1p0 = []
distance_g0p0 = []
distance_g1p1 = []

denominator = math.sqrt(w[0]**2 + w[1]**2)

for i in range (0, len(temp_g0)) :
	distance_g0p0.append(distance(mean_g0, temp_g0[i]))

for i in range (0, len(temp_g1)) :
	distance_g1p1.append(distance(mean_g1, temp_g1[i]))

for i in range (0, len(temp_g1)) :
	distance_g0p1.append(distance(mean_g0, temp_g1[i]))

for i in range (0, len(temp_g0)):
	distance_g1p0.append(distance(mean_g1, temp_g0[i]))

count_p0 = 0
count_p1 = 0
exclude_index = []

for i in range (0, len(distance_g0p0)):
	if distance_g0p0[i] > distance_g1p0[i] :
		exclude_index.append(index_g0[i])

for i in range (0, len(distance_g0p1)):
	if distance_g1p1[i] > distance_g1p0 :
		exclude_index.append(index_g1[i])

#for i in range(0, len(exclude_index)) :
#	print i,
#	print exclude_index[i]
print "exclude"
print exclude_index

exclude_index.reverse()
		
# remove exclude point
#for i in range(0, len(exclude_index)) :
#	print x.pop(exclude_index[i])
#	print y.pop(exclude_index[i])


#clf.fit(x, y)

#distance line to point
result = 0

temp_lp0 = []
temp_lp1 = []
index_lp0 = []
index_lp1 = []

result_dist0 = []
result_dist1 = []

for i in range(0, len(x)):
	if y[i] == 0 :
		temp_lp0.append(x[i])
		index_lp0.append(i)
	else :
		temp_lp1.append(x[i])
		index_lp1.append(i)

for i in range(0, len(temp_lp0)):
	numerator = abs(w[0]*temp_lp0[i][0] + w[1]*temp_lp0[i][1] + clf.intercept_[0])
	result_dist0.append(numerator / denominator)
	#dist_lp.append(numerator / denominator)

for i in range(0, len(temp_lp1)):
	numerator = abs(w[0]*temp_lp1[i][0] + w[1]*temp_lp1[i][1] + clf.intercept_[0])
	result_dist1.append(numerator / denominator)
	#dist_lp.append(numerator / denominator)


#result_dist0 = []
#result_dist1 = []

#cilassify group'
#for i in range(0, len(dist_lp)):
#	if y[i]==0 :
#		result_dist0.append(dist_lp[i])
#	else :
#		result_dist1.append(dist_lp[i])

#for i in range(0, len(temp_array2)):
#	print i,
#	print temp_array1[i]


#mean
mean_lp0 = np.mean(result_dist0)
mean_lp1 = np.mean(result_dist1)
print "mean_g0 : ",
print mean_lp0
print "mean_g1 : ",
print mean_lp1
#standard variance of distance
std_lp0 = np.std(result_dist0)
std_lp1 = np.std(result_dist1)

print std_lp0
print std_lp1

#under bound
dbound_lp0 = mean_lp0 - 1.96 * std_lp0
dbound_lp1 = mean_lp1 - 1.96 * std_lp1
print "underbound_0 : ",
print dbound_lp0
print "underbound_1 : ",
print dbound_lp1

#num of outlier
outlier_lp0 = []
outlier_lp1 = []
num_lp0 = 0
num_lp1 = 0
for i in range(0, len(temp_lp0)):
	if result_dist0[i] < dbound_lp0 :
		#outlier_lp0.append(index_lp0[i])
		exclude_index.append(index_lp0[i])
		num_lp0 += 1
for i in range(0, len(temp_lp1)):
	if result_dist1[i] < dbound_lp1 :
		#outlier_lp1.append(index_lp1[i])
		exclude_index.append(index_lp1[i])
		num_lp1 += 1
print ("num_lp0 : %d"%num_lp0)
#print outlier_lp0
print ("num_lp1 : %d"%num_lp1)
#print outlier_lp1



#data group mean
temp_x0, temp_x1 = [],[]
index_pp0, index_pp1 = [],[]

for i in range(0, len(x)):
	if y[i] == 0:
		temp_x0.append(x[i])
		index_pp0.append(i)
	else :
		temp_x1.append(x[i])
		index_pp1.append(i)

mean_pp0 = np.mean(temp_x0, axis=0)
mean_pp1 = np.mean(temp_x1, axis=0) 

print mean_pp0
print mean_pp1

# p2p : distance point to point

#def distance(p0, p1):
#	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

dist_pp0 = []
dist_pp1 = []
#index_pp0 = []
#index_pp1 = []

for i in range(0, len(temp_x0)):
	dist_pp0.append(distance(mean_pp0, temp_x0[i]))
for i in range(0, len(temp_x1)):
	dist_pp1.append(distance(mean_pp1, temp_x1[i]))

mean_distpp0 = np.mean(dist_pp0)
mean_distpp1 = np.mean(dist_pp1)

print "mean_distpp0 : ",
print mean_distpp0
print "mean_distpp1 : ",
print mean_distpp1

std_distpp0 = np.std(dist_pp0)
std_distpp1 = np.std(dist_pp1)

print std_distpp0
print std_distpp1

upb_distpp0 = mean_distpp0 + 1.96*std_distpp0
upb_distpp1 = mean_distpp1 + 1.96*std_distpp1

print "upper_boundpp0 : ",
print upb_distpp0
print "upper_boundpp1 : ",
print upb_distpp1
print " "

#num of outlier
outlier_pp0 = []
outlier_pp1 = []

numofout_pp0=0
numofout_pp1=0
for i in range(0, len(dist_pp0)):
	if dist_pp0[i] > upb_distpp0 :
		#outlier_pp0.append(index_pp0[i])
		exclude_index.append(index_pp0[i])
		numofout_pp0 += 1
for i in range(0, len(dist_pp1)):
	if dist_pp1[i] > upb_distpp0 :
		#outlier_pp1.append(index_pp1[i])
		exclude_index.append(index_pp1[i])
		numofout_pp1 += 1
print numofout_pp0
#print outlier_pp0
print numofout_pp1
#print outlier_pp1

exclude_index = list(set(exclude_index))
exclude_index.sort()
exclude_index.reverse()
print exclude_index

for i in range(0, len(exclude_index)) :
	print x.pop(exclude_index[i])
	print y.pop(exclude_index[i])

clf.fit(x, y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx1 = np.linspace(-12,5)
yy1 = a * xx - (clf.intercept_[0])/w[1]


#original
a1 = -0.89962
yy2 = a1 * xx - (clf.intercept_[0])/w[1]

print "===="
print a
print clf.intercept_[0]

x1 = np.asarray(x)
y1 = np.asarray(y)
x_outlier = np.asarray(x_outlier)
y_outlier = np.asarray(y_outlier)


# need not to modifying
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy -a * margin

plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx1, yy1, 'r-')
plt.plot(xx1, yy2, 'g-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)

plt.scatter(x1[:,0], x1[:,1], c=y, zorder=10, cmap=plt.cm.Paired)
plt.scatter(x_outlier[:,0], x_outlier[:,1], c=y, zorder=10, cmap=plt.cm.Paired)

plt.show()
