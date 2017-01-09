# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, Nawel Medjkoune <nawel.mdk@gmail.com>
# Licence: BSD 3 clause



from sklearn.model_selection import TimeSeriesSplit
import numpy as np                                                               



def linEx(y,yPred):
    i=0
    res=[]
    while i<len(y):
        
        res.append(np.exp(0.1*(y[i]-yPred[i]))-0.1*(y[i]-yPred[i])-1)
        #res.append(np.exp(0.15*(y-yPred))-0.15(y-yPred)-1)
        #res.append(np.exp(0.05*(y-yPred))-0.05(y-yPred)-1)
    return res.mean()

def crossVal(reg,X_train_scaled,set_Y_train):
    print "cross validation..."
        
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    score=[]

    tscv = TimeSeriesSplit(n_splits=3)        
    for train, test in tscv.split(X_train_scaled):
        X_train.append(train)
        X_test.append(test)

    for Ytrain, Ytest in tscv.split(set_Y_train):
        y_train.append(Ytrain)
        y_test.append(Ytest)

    scores = []
    for k in range(3):
        print len(X_train[k])
        print len(y_train[k])
        reg.fit(X_train[k] ,y_train[k] )
        y_pred=reg.predict(X_test[k] )

        score.append(linEx(y_test[k],y_pred))
    return score.mean()