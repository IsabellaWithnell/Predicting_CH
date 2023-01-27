### Large clone defined as VAF greater than 0.1
##Also will include code here for RF models to predict all size CH


### loading packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from IPython.display import Image 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.inspection import permutation_importance
import shap


from mlxtend.plotting import plot_decision_regions

import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

def get_results(truth, prediction, df, idx):
    
    y_test2 = truth.map({'No':0, 'Yes':1})
    y_pred2 = pd.Series(prediction).map({'No':0, 'Yes':1})


    acc = accuracy_score(y_test2, y_pred2)
    f1 = f1_score(y_test2, y_pred2)
    precision = precision_score(y_test2, y_pred2)
    recall = recall_score(y_test2, y_pred2)

    df.loc[idx,:] = [acc, f1, precision, recall]

    return df

df1 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanDNMT3Alargeclone01.csv')
df2 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanASXL1largeclone01.csv')
df3 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanATMlargeclone01.csv')
df4 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanCALRlargeclone01.csv')
df5 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanCBLlargeclone01.csv')
df6 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanGNASlargeclone01.csv')
df7 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanGNB1largeclone01.csv')
df8 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanJAK2largeclone01.csv')
df9 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanKMT2Alargeclone01.csv')
df10 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanKRASlargeclone01.csv')
df11 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanNF1largeclone01.csv')
df12 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanPPM1Dlargeclone01.csv')
df13 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanSF3B1largeclone01.csv')
df14 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanSRCAPlargeclone01.csv')
df15 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanSRSF2largeclone01.csv')
df16 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanSTAT3largeclone01.csv')
df17 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanTET2largeclone01.csv')
df18 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanTP53largeclone01.csv')
df19 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanTTNlargeclone01.csv')
df20 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanYLPM1largeclone01.csv')
df21 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanZBTB33largeclone01.csv')
df22 = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/JanZNF318largeclone01.csv')

#################################################################################### DNMT3A


data1 = df1

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

#################################################################################### ASXL1

data1 = df2

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)


## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### ATM


data1 = df3

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### CALR


data1 = df4

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### CBL


data1 = df5

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### GNAS


data1 = df6

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### GNB1


data1 = df7

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### JAK2


data1 = df8

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130,150,180],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8,12], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### KMT2A


data1 = df9

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### KRAS


data1 = df10

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### NF1


data1 = df11

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### PPM1D


data1 = df12

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### SF3B1


data1 = df13

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### SRCAP


data1 = df14

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### SRSF2


data1 = df15

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### STAT3


data1 = df16

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### TET2


data1 = df17

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### TP53


data1 = df18

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### TTN


data1 = df19

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### YLPM1


data1 = df20

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### ZBTB33


data1 = df21

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)





#################################################################################### ZNF318


data1 = df22

## Response Variable

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].map({'No':'No', 'Yes':'Yes'})

print(data1.iloc[:,-1].value_counts())

data1.iloc[:,-1] = data1.iloc[:,-1].astype('category')

## Change to category
for col in ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking']:
    
    data1[col] = data1[col].astype('category')
    

## Splitting data
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size=0.25, random_state=8)


## Set up results df
results_CH_large_01 = pd.DataFrame(index = ['Random Forest1', 'Random Forest2', 'Random Forest3', 
                               'Random Forest4', 'Random Forest5', 'Random Forest6'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

## Define categories train
def categories(X_train1):
   try:
      return X_train1[0]
   except TypeError:
      return "None"
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train1[variable],prefix = variable)        
   X_train1= pd.concat([X_train1, discarded], axis = 1)
   X_train1.drop([variable], axis = 1, inplace = True) 

## Define categories test
def categories(X_test1):
   try:
      return X_test1[0]
   except TypeError:
      return "None"
    
categorical_variables = ['Sex', 'Batch', 'Smoking_status','Alcohol_intake_frequency', 'Time_since_last_menstrual_period',
           'Alcohol_drink_status', 'Current_tobacco_smoking'] 

for variable in categorical_variables:
    
   discarded = pd.get_dummies(X_test1[variable],prefix = variable)        
   X_test1= pd.concat([X_test1, discarded], axis = 1)
   X_test1.drop([variable], axis = 1, inplace = True) 


parameters = {'max_features':['sqrt'], 'n_estimators':[50,60,70,80,90,100,110,120,130],
              'max_depth':range(2,12),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train1, y=y_train1)
rf_model = rf_class.best_estimator_

y_pred1 = rf_model.predict(X_test1)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred1).map({'No':0, 'Yes':1})

results = get_results(y_test1, y_pred1, results, 'Random Forest1')

print('The best parameters are {}'.format(rf_class.best_params_))

results_CH_large_01 = get_results(y_test1, y_pred1, results_CH_large_01, 'Random Forest1')
results_CH_large_01

train_probs = rf_model.predict_proba(X_train1)[:,1] 
probs = rf_model.predict_proba(X_test1)[:, 1]
train_predictions = rf_model.predict(X_train1)
print(f'Train ROC AUC Score: {roc_auc_score(y_train1, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test1, probs)}')
auc = metrics.roc_auc_score(y_test1, probs)

## Convert strings to ints to work with accuracy score functions
y_test1_ = y_test1.map({'No':0, 'Yes':1})
y_pred1_ = pd.Series(y_pred1).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred1_
fpr, tpr, _ = metrics.roc_curve(y_test1_,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

labels = ['No', 'Yes']
cm = confusion_matrix(y_test1, y_pred1, labels=labels)

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy

accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)


