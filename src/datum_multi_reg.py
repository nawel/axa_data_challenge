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
data=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/gpCalles.obj","rb"))
print "Loading the data to predict ..."
dataTest=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/testCalls.obj","rb"))
labels = data['CSPL_RECEIVED_CALLS'].unique()
#print "Creat Mean_calls"
#soustraire la moyenne
#mean_calls=data.groupby(['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute']).mean().reset_index()
#data['MEAN_calls'] = pd.merge(data, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['CSPL_RECEIVED_CALLS_y']
#data['CSPL_RECEIVED_CALLS']=data['CSPL_RECEIVED_CALLS']-data['MEAN_calls']
#dataTest['MEAN_calls'] = pd.merge(dataTest, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['MEAN_calls']



print "Loading the X and Y train ..."
set_X_train=[]
set_Y_train=[]
i=0
while i < len(data['cod_ASS_ASSIGNMENT'].unique()):
	set_X_train.append(data[[  'DAY_WE_DS','cod_ASS_ASSIGNMENT','year','month','day','hour','minute']][data['cod_ASS_ASSIGNMENT' ]==(i)])
	set_Y_train.append(data['CSPL_RECEIVED_CALLS'][data['cod_ASS_ASSIGNMENT' ]==(i)])
	i=i+1

print "Loading the X  test ..."
set_X_test=[]
i=0
while i < len(data['cod_ASS_ASSIGNMENT'].unique()):
	set_X_test.append(dataTest[[  'DAY_WE_DS','cod_ASS_ASSIGNMENT','year','month','day','hour','minute']][dataTest['cod_ASS_ASSIGNMENT' ]==(i)])
	i=i+1

i=0
listPred=[]
while i<len(set_X_train):
	print " Train et Predict the categorie : ",i
	reg=Regressor()
	reg.fit(set_X_train[i], set_Y_train[i])
	if(len(set_X_test[i])>0):
		listPred.append( reg.predict(set_X_test[i]))
	i=i+1


l=0
i=0
while l<len(set_X_test):
	if(len(set_X_test[l])>0):
		set_X_test[l]['CSPL_RECEIVED_CALLS'] =   listPred[i]
		i=i+1
	l=l+1



#on réassemble les valeurs de prédiction
resultPred= pd.concat(set_X_test,  ignore_index=True)

def make_submission(test, prediction, filename='/Users/ozad/Desktop/axa_data_challenge-master/submission.txt'):
	"""
	Create a submission file, 
	test: test dataset
	prediction: predicted values
	"""
	with open(filename, 'w') as f:
		f.write('DATE\tASS_ASSIGNMENT\tprediction\n')
		submission_strings = test['DATE'] + '\t' + test['ASS_ASSIGNMENT'] + '\t'+ prediction['CSPL_RECEIVED_CALLS'].astype(str)
		for row in submission_strings:
			f.write(row + '\n') 


#### write the submission
print "Write the submission ..."
make_submission(dataTest,resultPred)

print "End."
