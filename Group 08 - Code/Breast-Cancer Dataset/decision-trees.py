from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import numpy as np

'''
    LOAD CANCER DATASET
'''
cancer = load_breast_cancer()

''' 
    TRAIN-TEST SPLIT
    TEST_RATIO = 0.25
'''
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

'''
    GINI IMPURITY 
'''

'''
    THERE IS NOT LIMIT ON THE DEPTH OF DECISION TREE SO THE RESULTING MODEL MAY OVERFIT. ACCURACY OF TRAINING SET MIGHT GO 100%.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0)
clf.fit(X_train, y_train)
print("GINI Train Accuracy (no depth-limit) : ", clf.score(X_train, y_train))
print("GINI Test Accuracy (no depth-limit) : ", clf.score(X_test, y_test))

'''
    LIMITNG MAX-DEPTH OF THE TREE IS A PRE-PRUNING METHOD. THIS PREVENTS TREE FROM DEVELOPING COMPLETELY. SUBSEQUESNTLY,
    ACCURACY OF THE TREE ON TRAINING SET DROPS, BUT THE MODEL IS NO LONGER OVER-FITTED. THIS RESULTS IN INCREASED
    ACCURACY.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=0)
clf.fit(X_train, y_train)
print("GINI Train Accuracy (with depth-limit) : ", clf.score(X_train, y_train))
print("GINI Train Accuracy (with depth-limit) : ", clf.score(X_test, y_test))

'''
    DECISION TREE GRAPH
'''
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=cancer.feature_names, class_names=['malignant', 'benign'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision-tree-gini.png', view=True) 

'''
    ENTROPY IMPURITY 
'''

'''
    THERE IS NOT LIMIT ON THE DEPTH OF DECISION TREE SO THE RESULTING MODEL MAY OVERFIT. ACCURACY OF TRAINING SET MIGHT GO 100%.
'''
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X_train, y_train)
print("ENTROPY Train Accuracy (no depth-limit) : ", clf.score(X_train, y_train))
print("ENTROPY Test Accuracy (no depth-limit) : ", clf.score(X_test, y_test))

'''
    LIMITNG MAX-DEPTH OF THE TREE IS A PRE-PRUNING METHOD. THIS PREVENTS TREE FROM DEVELOPING COMPLETELY. SUBSEQUESNTLY,
    ACCURACY OF THE TREE ON TRAINING SET DROPS, BUT THE MODEL IS NO LONGER OVER-FITTED. THIS RESULTS IN INCREASED
    ACCURACY.
'''
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
clf.fit(X_train, y_train)
print("ENTROPY Train Accuracy (with depth-limit) : ", clf.score(X_train, y_train))
print("ENTROPY Train Accuracy (with depth-limit) : ", clf.score(X_test, y_test))

'''
    DECISION TREE GRAPH
'''
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=cancer.feature_names, class_names=['malignant', 'benign'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision-tree-entropy.png', view=True) 

'''
    FEATURE IMPORTANCE WILL TELL US HOW MUCH IMPORTANCE EACH FEATURE CARRIES
'''
n_features = cancer.data.shape[1]
plt.barh(range(n_features), clf.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()