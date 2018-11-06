import numpy as np
import pandas as pd
# import re
# import xgboost as xgb
# import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.tools as tls

# from sklearn import tree
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from IPython.display import Image as PImage
# from subprocess import check_call
# from PIL import Image, ImageDraw, ImageFont

data = pd.read_csv('data.csv')
original_dataset = data.copy()

full_dataset = [data]

for data in full_dataset:
    data['BUYING'] = data['BUYING'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)
    
    data['MAINT'] = data['MAINT'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)

    data['DOORS'] = data['DOORS'].map({'2': 0, '3': 1, '4': 2, '5more': 3}).astype(int)

    data['PERSONS'] = data['PERSONS'].map({'2': 0, '4': 1, 'more': 2}).astype(int)

    data['LUG_BOOT'] = data['LUG_BOOT'].map({'small': 0, 'med': 1, 'big': 2}).astype(int)

    data['SAFETY'] = data['SAFETY'].map({'low': 0, 'med': 1, 'high': 2}).astype(int)

    data['CLASS'] = data['CLASS'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)

feature_elements = ['CLASS']
features = data.drop(feature_elements, axis = 1)
np.savetxt(r'features.csv', features.values, fmt='%d', delimiter=',')

class_elements = ['BUYING', 'MAINT', 'DOORS', 'PERSONS', 'LUG_BOOT', 'SAFETY']
classes = data.drop(class_elements, axis = 1)
np.savetxt(r'class.csv', classes.values, fmt='%d', delimiter=',')
