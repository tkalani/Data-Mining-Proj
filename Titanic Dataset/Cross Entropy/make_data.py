import numpy as np
import pandas as pd
import re
import xgboost as xgb
# import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

print('hello world')

# Loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# train_final = open('trainfinal.txt', 'w+')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

# Showing overview of the train dataset
train.head(3)

# Copy original dataset in case we need it later when digging into interesting features
# WARNING: Beware of actually copying the dataframe instead of just referencing it
# "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')
original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values

# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

count = 0
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

# Feature selection: remove variables no longer containing relevant information

drop_elements1 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train1 = train.drop(drop_elements1, axis = 1)
np.savetxt(r'train_all_data.txt', train1.values, fmt='%d')

drop_elements1 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train1 = test.drop(drop_elements1, axis = 1)
np.savetxt(r'test_all_data.txt', train1.values, fmt='%d')

# drop_elements1 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Survived']
# train1 = train.drop(drop_elements1, axis = 1)
# np.savetxt(r'train.txt', train1.values, fmt='%d')

# drop_elements2 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'IsAlone', 'Title']
# train1 = train.drop(drop_elements2, axis = 1)
# np.savetxt(r'training_set_classes.txt', train1.values, fmt='%d')

# drop_elements1 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Survived']
# test1 = test.drop(drop_elements1, axis = 1)
# np.savetxt(r'testin.txt', test1.values, fmt='%d')

# drop_elements2 = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Has_Cabin', 'FamilySize', 'IsAlone', 'Title']
# test2 = test.drop(drop_elements2, axis = 1)
# np.savetxt(r'testclass.txt', test2.values, fmt='%d')

# for i in train:
#     print(i)

# import matplotlib.pyplot as plt
# import numpy as np

# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2*np.pi*t)
# plt.plot(t, s)

# plt.xlabel('time (s)')
# plt.ylabel('voltage (mV)')
# plt.title('About as simple as it gets, folks')
# plt.grid(True)
# plt.savefig("test.png")
# plt.show()