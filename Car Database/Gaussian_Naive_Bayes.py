# -*- coding: utf-8 -*-
"""Gaussian_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WFnn4k3Vgllywr-kwXYpp8334l6nWvX5
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('all_data.csv')
df.head()

x_data = df.drop(['CLASS'], axis=1)
y_label = df['CLASS']

X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.33, random_state=42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_test, y_pred))
