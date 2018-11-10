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
    map_dict = {'x': 0, 'o': 1, 'b': 2}
    data['TOP-LEFT-SQUARE'] = data['TOP-LEFT-SQUARE'].map(map_dict).astype(int)
    
    data['TOP-MIDDLE-SQUARE'] = data['TOP-MIDDLE-SQUARE'].map(map_dict).astype(int)

    data['TOP-RIGHT-SQUARE'] = data['TOP-RIGHT-SQUARE'].map(map_dict).astype(int)

    data['MIDDLE-LEFT-SQUARE'] = data['MIDDLE-LEFT-SQUARE'].map(map_dict).astype(int)

    data['MIDDLE-MIDDLE-SQUARE'] = data['MIDDLE-MIDDLE-SQUARE'].map(map_dict).astype(int)

    data['MIDDLE-RIGHT-SQUARE'] = data['MIDDLE-RIGHT-SQUARE'].map(map_dict).astype(int)

    data['BOTTOM-LEFT-SQUARE'] = data['BOTTOM-LEFT-SQUARE'].map(map_dict).astype(int)

    data['BOTTOM-MIDDLE-SQUARE'] = data['BOTTOM-MIDDLE-SQUARE'].map(map_dict).astype(int)

    data['BOTTOM-RIGHT-SQUARE'] = data['BOTTOM-RIGHT-SQUARE'].map(map_dict).astype(int)

    data['CLASS'] = data['CLASS'].map({'negative': 0, 'positive': 1,}).astype(int)

feature_elements = ['CLASS']
features = data.drop(feature_elements, axis = 1)
np.savetxt(r'features.csv', features.values, fmt='%d', delimiter=',')

class_elements = ['TOP-LEFT-SQUARE', 'TOP-MIDDLE-SQUARE', 'TOP-RIGHT-SQUARE', 'MIDDLE-LEFT-SQUARE', 'MIDDLE-MIDDLE-SQUARE', 'MIDDLE-RIGHT-SQUARE', 'BOTTOM-LEFT-SQUARE', 'BOTTOM-MIDDLE-SQUARE', 'BOTTOM-RIGHT-SQUARE']
classes = data.drop(class_elements, axis = 1)
np.savetxt(r'class.csv', classes.values, fmt='%d', delimiter=',')

data.to_csv('tic_tac_toe_all_data.csv')