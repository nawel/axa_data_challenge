import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np   
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor                           


data=pickle.load(open("../gpCalls.obj","rb"))
dataTest=pickle.load(open("../testCalls.obj","rb"))
set_X_train=[]
set_Y_train=[]


i=0
while i < len(data['cod_ASS_ASSIGNMENT'].unique()):
    set_X_train.append(data[[  'DAY_WE_DS','cod_ASS_ASSIGNMENT','year','month','day','hour','minute','PIQUE_NV','PIQUE_Bin']][data['cod_ASS_ASSIGNMENT' ]==(i)])
    set_Y_train.append(data['CSPL_RECEIVED_CALLS'][data['cod_ASS_ASSIGNMENT' ]==(i)])
    i=i+1

param_grid = {'learning_rate': [1,0.1, 0.05, 0.02, 0.01,0.001,0.0001],
              'n_estimators':[10,500, 1000,1500, 2000,2500,4000],
              'max_depth':[1,5, 10, 15],
              'min_samples_leaf':[1,10, 20, 40, 100],
              'random_state':[10,20,30,40,50]
              }

est = GradientBoostingRegressor()
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(set_X_train[0], set_Y_train[0])

# best hyperparameter setting
print gs_cv.best_params_
print gs_cv.best_score_