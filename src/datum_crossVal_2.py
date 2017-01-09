# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, Nawel Medjkoune <nawel.mdk@gmail.com>
# Licence: BSD 3 clause

import pickle
import graphlab as gl
import pandas as pd
from datetime import datetime, timedelta
import numpy as np                                                               
from utils import make_submission
from regressor import Regressor
import math
from sklearn.preprocessing import StandardScaler
from crossVal import crossVal

print "Start"
print "Loading the data train ..."

data=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/gpCalls.obj","rb"))
print "Loading the data to predict ..."
dataTest=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/testCalls.obj","rb"))
labels = data['CSPL_RECEIVED_CALLS'].unique()
#print "Creat Mean_calls"
#soustraire la moyenne
#mean_calls=data.groupby(['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute']).mean().reset_index()
#data['MEAN_calls'] = pd.merge(data, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['CSPL_RECEIVED_CALLS_y']
#data['CSPL_RECEIVED_CALLS']=data['CSPL_RECEIVED_CALLS']-data['MEAN_calls']
#dataTest['MEAN_calls'] = pd.merge(dataTest, mean_calls, how='left', on=['ASS_ASSIGNMENT', 'DAY_WE_DS','hour', 'minute'])['MEAN_calls']


#Defining features
features=['DATE','DAY_WE_DS', 'year','month','day','hour','minute', 'WEEK_END','cod_ASS_ASSIGNMENT']
features_train=['DAY_WE_DS', 'year' ,'month','day','hour','minute','WEEK_END']
    
print "Loading dates..."

dates=np.sort((dataTest[dataTest['hour']==0])[dataTest['minute']==0]['DATE'].unique())
dates_a_predire=[]
for i in range(12):
	dates_a_predire.append(dates[i*7])	
print "dates to predict " , dates_a_predire

incremental_prediction=[]
score_global  =[]                       
for i_date in range(len(dates_a_predire)):
    print "Selecting data from dates to predict...", i_date

    sub_data=data[data['DATE'] < dates_a_predire[i_date]]
    sub_test=dataTest[dataTest['DATE'] >= dates_a_predire[i_date]]
    if(i_date!=len(dates_a_predire)-1):
        sub_test=sub_test[dataTest['DATE'] < dates_a_predire[i_date+1]]

    print "Loading the X and Y train ..."
    set_X_train=[]
    set_Y_train=[]

    i=0
    while i < len(sub_data['cod_ASS_ASSIGNMENT'].unique()):
        set_X_train.append(sub_data[features][sub_data['cod_ASS_ASSIGNMENT']==(i)])
        set_Y_train.append(sub_data['CSPL_RECEIVED_CALLS'][sub_data['cod_ASS_ASSIGNMENT']==(i)])
        i=i+1
        
    print "Loading the X test ..."
    set_X_test=[]
    i=0
    while i < len(sub_data['cod_ASS_ASSIGNMENT'].unique()):
        set_X_test.append(sub_test[features][sub_test['cod_ASS_ASSIGNMENT' ]==(i)])
        i=i+1

    i=0
    listPred=[]
    score=[]
    while i<len(set_X_train):
        scaler = StandardScaler().fit(set_X_train[i][features_train])
        X_train_scaled = scaler.transform(set_X_train[i][features_train])
        print " Train et Predict the categorie : ",i
        reg=Regressor()
        #reg.fit(X_train_scaled, set_Y_train[i])


        #### Cross validation
        #print "Cross validation ...", i
        #loo = cross_validation.LeaveOneOut(len(y_df))
        #loo=10
        #scores = cross_validation.cross_val_score(reg, X_train_scaled, set_Y_train[i], scoring='neg_mean_squared_error', cv=loo,)
        #print "The score mean of cross validation : ", scores.mean()
        #score_cv_global.append(scores.mean())
        
        
        ##### cross validation :
        
        score.append(crossVal(reg,X_train_scaled,set_Y_train[i]))


        
        """ 
        if(len(set_X_test[i])>0):            
            X_test_scaled = scaler.transform(set_X_test[i][features_train])
            listPred.append( reg.predict(X_test_scaled))
            
        """
            

        i=i+1

    
    score_global.append(score.mean())
    """ 

    l=0
    i=0
    while l<len(set_X_test):
    if(len(set_X_test[l])>0):
        set_X_test[l]['CSPL_RECEIVED_CALLS'] =   listPred[i]
        i=i+1
    l=l+1
    """


    #on réassemble les valeurs de prédiction
    resultPred= pd.concat(set_X_test)
    resultPred=resultPred.sort_index()
    incremental_prediction.append(resultPred)
print "score global = ",score_global.mean()

print("Merging incremental learning...")
resultPred_final=pd.concat(incremental_prediction)
resultPred_final=resultPred_final.sort_values(by=['DATE', 'cod_ASS_ASSIGNMENT'])

print("Make every prediction positif, ceil it ...")
resultPred_final['CSPL_RECEIVED_CALLS']=resultPred_final['CSPL_RECEIVED_CALLS'].apply(lambda x: x*(x>0))
#resultPred_final['CSPL_RECEIVED_CALLS']=resultPred_final['CSPL_RECEIVED_CALLS'].apply(lambda x: 2.5*x)
resultPred_final['CSPL_RECEIVED_CALLS']=resultPred_final['CSPL_RECEIVED_CALLS'].apply(lambda x: math.ceil(x))

print "Write the submission ..."
make_submission(dataTest,resultPred_final)
print "End."

