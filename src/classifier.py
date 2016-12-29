# coding: utf-8
# Author:  Daro HENG <daro.heng@u-psud.fr>, 
# Licence: BSD 3 clause



from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 10
        self.n_estimators = 400
        self.clf = Pipeline([
            #('pca', PCA(n_components=self.n_components)), 
            #('pca',  KernelPCA(n_components=21,kernel="rbf",remove_zero_eig=True,gamma=0.0004)),
            
            ####### Basic algo :  Random Forest Classifier
            #('clf', RandomForestClassifier(n_estimators=self.n_estimators, random_state=42))
            
            ####### GradientBoost
            #('clf', GradientBoostingClassifier(n_estimators=self.n_estimators,max_depth=1000,subsample=0.85,
            #       max_features='log2',random_state=42))
                    
                 
           ####### AdaBoost avec DecisionTreeClassifier
            ('clf', AdaBoostClassifier(
                   base_estimator=DecisionTreeClassifier(max_depth=8),
                   n_estimators=1000,
                   learning_rate=1.6,
                   random_state=42,
                   algorithm='SAMME',
               ))
                
            #('clf', QuadraticDiscriminantAnalysis()),
            #('clf', LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=10, store_covariance=True, tol=0.01))
            
                
                
            ####### AdaBoost avec GradientBoostingClassifier
            #('clf', AdaBoostClassifier(
            #    base_estimator=GradientBoostingClassifier(n_estimators=10000,max_depth=10,
            #       subsample=0.5,
            #       max_features='log2'),
            #    n_estimators=500,
            #    learning_rate=1,
            #    random_state=42,
            #    algorithm='SAMME',
            #))    
            
                
            ####### AdaBoost avec SVC
            #('clf', AdaBoostClassifier(
            #    svc(probability=True, kernel='linear'),
            #    n_estimators=10,
            #    learning_rate=1,
            #    random_state=42,
            #    algorithm='SAMME',
            #))
                
                
            ####### AdaBoost avec GaussianNB
            #('clf', AdaBoostClassifier(
            #    GaussianNB(),
            #  n_estimators=10,
            #  learning_rate=1,
            #  random_state=42,
            #  algorithm='SAMME',
            #))

        
            ####### AdaBoost avec SGDClassifier ou SVC
            #('clf', AdaBoostClassifier(
            #SGDClassifier(loss='log'),
            #n_estimators=10000,
            #learning_rate=1.6,
            #random_state=42,
            #algorithm='SAMME',
             #))
                

            ####### Bayesien 
            #('clf',  GaussianNB())
            
            ####### Multinomial NB
            #('clf', MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None))
                
            ####### SGD 
            # ('clf',  SGDClassifier(loss='log')
            #('clf',SGDClassifier())


        ])        



    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)