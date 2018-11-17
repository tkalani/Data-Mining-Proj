import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine

'''
    WINE DATA SET
'''
wine = load_wine()

'''
    TRAIN - TEST SPLIT, TEST_RATIO = 0.25
'''
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, stratify=wine.target, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))