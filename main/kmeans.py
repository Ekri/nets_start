from sklearn.cluster import KMeans
from sklearn import datasets
from itertools import cycle, combinations
import matplotlib.pyplot as pl

iris = datasets.load_iris()
X = iris.data
km = KMeans(n_clusters=3)
km.fit(X)

predictions = km.predict(iris.data)

colors = cycle('rgb')
labels = ["Cluster 1", "Cluster 2", "Cluster 3"]
targets = range(len(labels))

feature_index = range(len(iris.feature_names))
feature_names = iris.feature_names
combs = combinations(feature_index, 2)


