import numpy as np
import pandas as pd
import sklearn.ensemble
import pickle
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import copy
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_absolute_error
import sklearn.linear_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVR

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
train1 = copy.copy(train)
train = train[train['price'] != 1000000]
train = train.replace(np.nan, -1)
test = test.replace(np.nan, -1)
COLUMNS = ['area', 'street_id', 'rooms', 'kw11']
new_train = train[COLUMNS]
new_test = test[COLUMNS]


kmeans1 = KMeans(n_clusters=26, random_state=0).fit(new_train.values)
lab1 = kmeans1.labels_
new_train['labels'] = lab1
y = train['price'].values

tr_lab = new_train[COLUMNS]
y_lab = lab1
pred_lab = sklearn.ensemble.RandomForestRegressor()
pred_lab.fit(tr_lab, y_lab)

lab_for_test = pred_lab.predict(new_test)

new_test['labels'] = lab_for_test

X = new_train.values
Xt = new_test.values

mdl = sklearn.ensemble.RandomForestRegressor()

mdl.fit(X, y)

preds = mdl.predict(Xt)

test['price'] = preds

test[['id', 'price']].to_csv('malo_prizn_sub+25.csv', index=False)

