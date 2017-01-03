# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, 
# Licence: BSD 3 clause

import pickle
import graphlab as gl
import pandas as pd
from datetime import datetime, timedelta
import numpy as np                                                               
from utils import make_submission
from regressor import Regressor
from sklearn import cross_validation

print "Start"
print "Loading the data train ..."
data=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/callsData_DataFrame.obj","rb"))

print "Loading the data to predict ..."
dataTest=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/test_DataFrame.obj","rb"))

labels = data['CSPL_RECEIVED_CALLS'].unique()
#### Load the X train and Y train
Y_train=data['CSPL_RECEIVED_CALLS'][:300000]
X_train=data[  [  'DAY_WE_DS',    'cod_ASS_ASSIGNMENT','year','month','day','hour','minute']][:300000]
X_test=dataTest[[  'DAY_WE_DS',    'cod_ASS_ASSIGNMENT','year','month','day','hour','minute']]

Y_train=np.array(Y_train)
X_train=np.array(X_train)
X_test=np.array(X_test)


#### Creation of regressor 
reg=Regressor()


#### Cross validation
print "Cross validation ..."
#loo = cross_validation.LeaveOneOut(len(y_df))
loo=10
scores = cross_validation.cross_val_score(reg, X_train, Y_train, scoring='mean_squared_error', cv=loo,)
print "The score mean of cross validation : "
print scores.mean()

#### fit 
print "Fit ..."
reg.fit(X_train, Y_train)


#### Prediction
print "Prediction ..."
Y_pred = reg.predict(X_test)

#### write the submission
print "Write the submission ..."
make_submission(dataTest,Y_pred)

print "End."