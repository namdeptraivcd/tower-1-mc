from __future__ import print_function
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets



iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#information about dataset
'''
print("Number of class: %d" %len(np.unique(iris_y)))
print("Number of data points: %d" %len(iris_y))

X0 = iris_X[iris_y ==0, :]
print ('\nsamples from class 0:\n', X0[:5,:])
X1 = iris_X[iris_y == 1, :]
print("\nSamples from class 1:\n", X1[:5, :])
X2 = iris_X[iris_y == 2, :]
print("\nSamples from class 2:\n", X2[:5, :])

'''



X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)
print('\nTraining set size: %d' %len(y_train))
print('\nTest set size: %d' %len(y_test))


clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p=2, weights = "distance")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("|The 20 first output:")
print("The predicted output: ", y_pred[:20])
print("The true output: ", y_test[:20])

#Evaluation
from sklearn.metrics import accuracy_score
print("\nAccuracy of 10NN :%.2f%%" %(accuracy_score(y_test, y_pred) * 100))