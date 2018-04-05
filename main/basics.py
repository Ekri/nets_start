import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

array = np.array(list(zip(iris.data, iris.target)))[0:10]
# np.column_stack((iris.data, iris.target))[0:10]
""" Good alternative when do not want to use python functions """

print(array)
# print(iris.DESCR)
print(iris.target_names)


