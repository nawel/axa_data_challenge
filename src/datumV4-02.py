
# coding: utf-8

# In[1]:

import pandas as pd
from datetime import datetime, timedelta
from sklearn import linear_model
from numpy import array
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
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# In[2]:

########################################### Preprocessing #################################
######load the data, only the col in "usecols"
print "Loading the data from csv file... "
data = pd.read_csv('../train_2011_2012_2013.csv', parse_dates = ['DATE'], sep = ';',  engine='c', usecols=['DATE','DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS'])
print "End."

n = data.shape[0]
data = data.sort_values(by=['DATE'])
labels = data['CSPL_RECEIVED_CALLS'].unique()


# In[ ]:


######Loading the data in submission.txt
print ("Loading the data submission...")
file = open('../submission.txt','r')
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


# In[ ]:

######Transform the categories : ASS_ASSIGNMENT to ID
# a changer avec la commande :
#X2=pandas.Categorical.from_array(data['ASS_ASSIGNMENT']).codes[:pos1]
#y2 = np.array(col_received[:pos1])

print "Transform the categories : ASS_ASSIGNMENT to ID..."
m = len(submission)
delta = timedelta(minutes=30)
ass_num = {}
cont_ass = 0
for i in range(0,n):
    ass = data['ASS_ASSIGNMENT'][i]
    if ass not in ass_num:
        ass_num[ass] = cont_ass
        cont_ass += 1
print "End."
########################################### End Preprocessing #################################


# In[ ]:


##################################### Load Data #########################################################
col_date = data['DATE']
col_assignment = data['ASS_ASSIGNMENT']
col_received = data['CSPL_RECEIVED_CALLS']


###### load the data for train
print "Loading data for train..."
X = []
y = []
pos1, pos2 = 0, 0
d = submission[pos2][0]
d_str = pd.tslib.Timestamp("%d-%02d-%02d %02d:%02d:00.000" % (d.year, d.month, d.day, d.hour, d.minute))
while pos1 < n and col_date[pos1] <= d_str:
        tmp=col_assignment[pos1].decode("utf-8")
        ass_assignment = ass_num[tmp.encode("utf-8")]
        X.append([ass_assignment])
        y.append([ col_received[pos1] ])
        pos1 += 1
print "End."


###### load the data for test
#we test on the first test : 28/12/2012 to 3/01/2012
print "Loading data for validation..."
posTest=6820
d_test = submission[posTest][0]
X_test = []
pos3=0
while pos3 < m and submission[pos3][0] < d_test:
    ass_assignment = submission[pos3][1]
    X_test.append([ ass_num[ass_assignment] ])
    pos3 += 1
print "End."

##################################### End Load Data #########################################################


# In[ ]:


##################################### Classifier training ###########################################################    
clf = linear_model.SGDClassifier()
clf2 = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=8),
                n_estimators=10000,
                learning_rate=1.6,
                random_state=42,
                algorithm='SAMME',
            )
clf3=Classifier()
print "Training..."
if len(X) > 0:
 X = array(X)
 y = array(y).ravel()
 clf.partial_fit(X,y,classes=labels)
 #clf2.fit(X,y)
print "End."


# In[ ]:




##################################### test #############################################################   
######## predict y
print "Prediction..."
if len(X_test) > 0:
    X_test = array(X_test)
    y_pred = clf.predict(X_test)
print pos3
print "End."

####### affichage 
for i in range(pos3):
    print "%d-%d-%d %d:%d:00.000\t%s\t%d" % (submission[i][0].year,
        submission[i][0].month, submission[i][0].day,
        submission[i][0].hour, submission[i][0].minute, submission[i][1],
        y_pred[i])



#TODO :
#-> faire les préprocessing ( voir pySpark...)
#-> faire varier les dimension ( c'est à dire : ajouter ou enlever les colonnes dans X)
#-> tester les différentes modèles de classifier et regression
#-> implémenter le cross validation
#-> écrire les résultats dans le fichier submit.txt 



# In[ ]:



