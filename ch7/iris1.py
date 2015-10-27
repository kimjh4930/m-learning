import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()

x=iris.data
y=iris.target

pca = PCA(n_components=4)

reduced_x = pca.fit_transform(x)

print(pca.explained_variance_ratio_)
