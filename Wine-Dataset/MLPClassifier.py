from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
    LOAD WINE DATA SET
'''
wine = load_wine()

'''
    TRAIN - TEST SPLIT
    TEST_RATIO = 0.25
'''
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, stratify=wine.target, random_state=67)

'''
    MLP CLASSIFIER
    1000 EPOCHS ( 200 MAX-ITER CONVERGENCE ERROR )
    NO SCALING OF DATA
'''
mlp = MLPClassifier(max_iter=1000, random_state=92)
mlp.fit(X_train, y_train)
print("Training Accuracy (before scaling) : ", mlp.score(X_train, y_train))
print("Test Accuracy (before scaling) : ", mlp.score(X_test, y_test))

'''
    ACCURACY ON TRAINING AND TEST DATASETS IS NOT GOOD ENOUGH WHICH MAY BE DUE TO SCALING OF DATA.
    FEATURES OF EACH SAMPLE OF DATASET ARE NOT AT SAME SCALE.
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

'''
    ACCURACIES AFTER SCALING OF DATA.
    PERFORMANCE IS MUCH BETTER WITH SCALED PARAMETERS
'''
mlp = MLPClassifier(max_iter=1000, random_state=92)
mlp.fit(X_train_scaled, y_train)
print("Training Accuracy (after scaling) : ", mlp.score(X_train_scaled, y_train))
print("Test Accuracy (after scaling) : ", mlp.score(X_test_scaled, y_test))

'''
    COLORBAR
'''
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='None', cmap='GnBu')
plt.yticks(range(13), wine.feature_names)
plt.xlabel('Columns is weight matrix')
plt.ylabel('Input Feature')
plt.colorbar()