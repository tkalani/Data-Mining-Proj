from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import numpy as np

'''
    LOAD WINE DATASET
'''
wine = load_wine()

''' 
    TRAIN-TEST SPLIT
    TEST_RATIO = 0.25
'''
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, stratify=wine.target, random_state=92)

'''
    GINI IMPURITY 
'''

'''
    THERE IS NOT LIMIT ON THE DEPTH OF DECISION TREE SO THE RESULTING MODEL MAY OVERFIT. ACCURACY OF TRAINING SET MIGHT GO 100%.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=57)
clf.fit(X_train, y_train)
print("GINI Train Accuracy (no depth-limit) : ", clf.score(X_train, y_train))
print("GINI Train Accuracy (no depth-limit) : ", clf.score(X_test, y_test))

'''
    LIMITNG MAX-DEPTH OF THE TREE IS A PRE-PRUNING METHOD. THIS PREVENTS TREE FROM DEVELOPING COMPLETELY. SUBSEQUESNTLY,
    ACCURACY OF THE TREE ON TRAINING SET DROPS, BUT THE MODEL IS NO LONGER OVER-FITTED. THIS RESULTS IN INCREASED
    ACCURACY.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=57)
clf.fit(X_train, y_train)
print("GINI Train Accuracy (with depth-limit) : ", clf.score(X_train, y_train))
print("GINI Train Accuracy (with depth-limit) : ", clf.score(X_test, y_test))

'''
    DECISION TREE GRAPH
'''
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision-tree-gini.png', view=True)  

'''
    GINI IMPURITY 
'''

'''
    THERE IS NOT LIMIT ON THE DEPTH OF DECISION TREE SO THE RESULTING MODEL MAY OVERFIT. ACCURACY OF TRAINING SET MIGHT GO 100%.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=57)
clf.fit(X_train, y_train)
print("GINI Train Accuracy (no depth-limit) : ", clf.score(X_train, y_train))
print("GINI Train Accuracy (no depth-limit) : ", clf.score(X_test, y_test))

'''
    LIMITNG MAX-DEPTH OF THE TREE IS A PRE-PRUNING METHOD. THIS PREVENTS TREE FROM DEVELOPING COMPLETELY. SUBSEQUESNTLY,
    ACCURACY OF THE TREE ON TRAINING SET DROPS, BUT THE MODEL IS NO LONGER OVER-FITTED. THIS RESULTS IN INCREASED
    ACCURACY.
'''
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=57)
clf.fit(X_train, y_train)
print("ENTROPY Train Accuracy (with depth-limit) : ", clf.score(X_train, y_train))
print("ENTROPY Train Accuracy (with depth-limit) : ", clf.score(X_test, y_test))

'''
    DECISION TREE GRAPH
'''
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision-tree-entropy.png', view=True)

'''
    FEATURE IMPORTANCE WILL TELL US HOW MUCH IMPORTANCE EACH FEATURE CARRIES
'''
n_features = wine.data.shape[1]
plt.barh(range(n_features), clf.feature_importances_, align='center')
plt.yticks(np.arange(n_features), wine.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()