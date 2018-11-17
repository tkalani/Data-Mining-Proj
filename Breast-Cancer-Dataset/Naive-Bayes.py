import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer

'''
    LOAD CANCER DATA SET
'''
cancer = load_breast_cancer()

'''
    TRAIN - TEST SPLIT, TEST_RATIO = 0.25
'''
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))