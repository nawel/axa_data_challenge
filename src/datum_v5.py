# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, 
# Licence: BSD 3 clause


import pickle
import graphlab as gl
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import make_submission
from classifier import Classifier


####Load the data set
print ("Loading data set...")
data=pickle.load(open("/Users/ozad/Desktop/axa_data_challenge-master/callsData_DataFrame.obj","rb"))
labels = data['CSPL_RECEIVED_CALLS'].unique()
print ("End.")


####Load the data  to train
print ("Loading data train...")
y_df=data['CSPL_RECEIVED_CALLS'][:300000]
X_df=data[  ['WEEK_END',  'DAY_WE_DS',  'TPER_TEAM',  'ASS_ASSIGNMENT',  'ASS_DIRECTORSHIP','ASS_PARTNER','year','month','day','hour','minute']][:300000]
print ("End.")


####Load the data of submission
print ("Loading the data submission...")
file = open('/Users/ozad/Desktop/AXA/submission.txt','r')
lines = [line.rstrip('\n') for line in file]
lines = lines[1:]
submission = []
for line in lines:
    tokens = line.split()
    date1 = tokens[0].split("-")
    date2 = tokens[1].split(":")
    date = datetime(int(date1[0]),int(date1[1]),int(date1[2]),int(date2[0]),int(date2[1]))
    ass_assignment = ' '.join(tokens[2:-1])
    submission.append((date, ass_assignment))
print "End."


####Training
#PS : Pour changer de classifier il faut aller changer dans le fichier "classifier.py"
clf=Classifier()
if len(X_df) > 0:
    print "Training..."
    X = np.array(X_df)
    y = np.array(y_df).ravel()
    clf.fit(X,y)
print "End."


####Prediction
print "Prediction..."
if len(X_test) > 0:
    X_test = np.array(X_test)
    y_pred = clf.predict(X_test)
print pos3
print "End."


#### Wirte the prediction
make_submission(X_test,y_pred)

