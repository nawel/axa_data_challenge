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

print "Start"
print "Loading the data train ..."
data=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/callsData_DataFrame.obj","rb"))

print "Loading the data to predict ..."
dataTest=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/test_DataFrame.obj","rb"))

labels = data['CSPL_RECEIVED_CALLS'].unique()

#soustraire la moyenne
mean_calls=data.groupby(['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute']).mean().reset_index()
data['MEAN_calls'] = pd.merge(data, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['CSPL_RECEIVED_CALLS_y']
data['CSPL_RECEIVED_CALLS']=data['CSPL_RECEIVED_CALLS']-data['MEAN_calls']
dataTest['MEAN_calls'] = pd.merge(dataTest, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['MEAN_calls']

#### Load the X train and Y train
Y_train=data['CSPL_RECEIVED_CALLS']
X_train=data[  [  'DAY_WE_DS','cod_ASS_ASSIGNMENT','year','month','day','hour','minute']]
X_test=dataTest[[  'DAY_WE_DS','cod_ASS_ASSIGNMENT','year','month','day','hour','minute']]

Y_train=np.array(Y_train)
X_train=np.array(X_train)
X_test=np.array(X_test)

#### fit 
print "Fit ..."
reg=Regressor()
reg.fit(X_train, Y_train)


#### Prediction
print "Prediction ..."
Y_pred = reg.predict(X_test)
#rajouter le mean
Y_pred=Y_pred+dataTest['MEAN_calls']
#### write the submission
print "Write the submission ..."
make_submission1(dataTest,Y_pred)

print "End."
