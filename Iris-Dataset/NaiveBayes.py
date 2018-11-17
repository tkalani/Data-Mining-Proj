import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

'''
    LOAD IRIS DATA SET
'''
iris = load_iris()

'''
    TRAIN - TEST SPLIT
'''
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=0)

'''
    GAUSSIAN NAIVE BAYES CLASSIFIER
'''
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))