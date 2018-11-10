import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('all_data.csv')
df.head()

x_data = df.drop(['CLASS'], axis=1)
y_label = df['CLASS']

X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

