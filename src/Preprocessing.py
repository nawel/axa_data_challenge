import graphlab as gl
from graphlab.toolkits.feature_engineering import NumericImputer
import numpy as np
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import os

print ('Loading the data ..')
url = '/home/fichette/Documents/PROJETS/AXA/train_2011_2012_2013.csv'
#Read only some columns: the green ones in the sheet (see drive)
callsData = gl.SFrame.read_csv(url, delimiter=';', header=True,
                               usecols=['DATE','DAY_OFF','WEEK_END','DAY_WE_DS','TPER_TEAM','TPER_HOUR',
                                        'ASS_ASSIGNMENT','ASS_DIRECTORSHIP','ASS_PARTNER','ASS_POLE',
                                        'CSPL_CALLSOFFERED' , 'CSPL_NOANSREDIR','CSPL_ACDCALLS','CSPL_ABNCALLS',
                                        'CSPL_DISCCALLS','CSPL_MAXINQUEUE','CSPL_DEQUECALLS', 
                                        'CSPL_SERVICELEVEL','CSPL_ACCEPTABLE', 'CSPL_MAXSTAFFED', 
                                        'CSPL_ABANDONNED_CALLS', 'CSPL_CALLS','CSPL_RECEIVED_CALLS'])
#get Integer columns
cols = callsData[int].column_names()
#Remove columns where the values are binary 
cols = [c for c in cols if not len(callsData[c].unique()) <=2]
print ('Replacing missing values in continious variable with the mean ..')

imputer = gl.feature_engineering.create(callsData,
             NumericImputer(features = cols, strategy = 'mean'))
callsData = imputer.transform(callsData)

print ('Replacing missing values in binary variable with the most probable value ..')
bcols = [c for c in cols if not len(callsData[c].unique()) ==2]

for c in bcols:
    mask = callsData[c] == None
    if len(mask[mask==1]) > 0:
        imputer_str = gl.feature_engineering.CategoricalImputer(feature=c)
        transformed_sf = imputer_str.fit_transform(callsData)
        newc = 'predicted_feature_' + c
        callsData[c] = transformed_sf[newc]

print ('Replacing missing values in binary variable with the most probable value ..')
strCols = callsData[str].column_names()

for c in strCols :
    mask = callsData[c] == None
    if len(mask[mask==1]) > 0:
        imputer_str = gl.feature_engineering.CategoricalImputer(feature=c)
        transformed_sf = imputer_str.fit_transform(callsData)
        newc = 'predicted_feature_' + c
        callsData[c] = transformed_sf[newc]

print ('Converting to Pandas DataFrame')
X = callsData.to_dataframe()

print ('Replacing categorical values to IDs using Pandas')

for c in strCols:
    X[c]= (X[c].astype('category')).cat.codes

pickle.dump(X, open(os.getcwd()+"/callsData_DataFrame.obj", "wb"))

print ('Applying PCA...')

pca = PCA(n_components=10)
pca.fit(X)
print('PCA Results: ')
print (pca.explained_variance_ratio_) 
pickle.dump(pca, open(os.getcwd()+"/PCA.obj", "wb"))

