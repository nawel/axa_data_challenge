# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, 
# Licence: BSD 3 clause


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor                           
from sklearn.decomposition import PCA                                            
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator    
from sklearn.neural_network import MLPRegressor
import numpy as np                                                               
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor




class Regressor(BaseEstimator):                                                  
    def __init__(self):                                                          
        self.n_components = 8                                                 
        self.n_estimators = 40                                                  
        self.learning_rate = 0.2                                                
        self.reg = Pipeline([                                      
               #('pca', PCA(n_components=self.n_components)),        
                            
                ###### kernel PCA
                #('pca',  KernelPCA(n_components=21,kernel="rbf",remove_zero_eig=False)),


                ###### Gradient Boosting Regressor
                #('reg', GradientBoostingRegressor(n_estimators=self.n_estimators,learning_rate=self.learning_rate,random_state=42))

                
                ###### SVR
                #('reg',SVR(C=1.0, cache_size=200, coef0=0.0, degree=6, epsilon=0.2, gamma='auto',kernel='rbf', max_iter=5, shrinking=True, tol=0.001, verbose=False))
                
                ##### linear model
                #('reg',linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=5, normalize=False))
               
                ##### MLP Regressor
                #('reg',MLPRegressor(hidden_layer_sizes=(7,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                #learning_rate='constant', learning_rate_init=1, power_t=0.5, max_iter=1000, shuffle=True, random_state=42, 
                #tol=0.0001, verbose=False, warm_start=False, momentum=0.09, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                #beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                
               
                ##### Neighbors Regressor
                #('reg',RadiusNeighborsRegressor(radius=5.0))
                
                ##### PLS Regression
                #('reg', PLSRegression(copy=True, max_iter=500, n_components=2, scale=True,tol=1e-06))
                    
                ##### Decision Tree Regressor
                #('reg',DecisionTreeRegressor(max_depth=5))
                    
                ##### Random Forest Regressor
                ('reg',RandomForestRegressor(n_estimators=100))
                
                ##### Ada Boost Regressor avec Decision Tree Regressor
                #('reg', AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),
                #        n_estimators=3000, random_state=42))
            ])                                                                   
                                                                                 
    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
