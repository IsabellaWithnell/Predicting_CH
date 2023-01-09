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
import pyreadr
import os



def get_results(truth, prediction, df, idx):
    
    y_test2 = truth.map({'No':0, 'Yes':1})
    y_pred2 = pd.Series(prediction).map({'No':0, 'Yes':1})
    acc = accuracy_score(y_test2, y_pred2)
    f1 = f1_score(y_test2, y_pred2)
    precision = precision_score(y_test2, y_pred2)
    recall = recall_score(y_test2, y_pred2)

    df.loc[idx,:] = [acc, f1, precision, recall]

    return df

### load here the gene stratifiied dataset e.g. JAK2, ASXL1...
df = pd.read_csv('/rds/general/user/iw413/home/Summerproject/outputs/python_geneaASXL1oct.csv')

df.dtypes
data = df
eid = data.iloc[:, :1]
data = df.iloc[: , 1:]

    
for col in ['Smoking_status' , 'Batch' , 'Current_tobacco_smoking' ,
                'Alcohol_intake_frequency'  , 'Time_since_last_menstrual_period' , 'Alcohol_drink_status'  ,'Sex']:
    data[col] = data[col].astype('category')
    

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.25, random_state=8)





## Set up results df --> only using random forest here

results = pd.DataFrame(index = ['Decision Tree', 'Random Forest', 'Gradient Boost'], 
                       columns = ['accuracy', 'f1', 'precision', 'recall'])

def categories(X_train):
   try:
      return X_train[0]
   except TypeError:
      return "None"
categorical_variables = [ 'Smoking_status' , 'Batch', 'Current_tobacco_smoking' ,
                'Alcohol_intake_frequency'  , 'Time_since_last_menstrual_period' , 'Alcohol_drink_status'  ,'Sex'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_train[variable],prefix = variable)        
   X_train= pd.concat([X_train, discarded], axis = 1)
   X_train.drop([variable], axis = 1, inplace = True) 

X_train


def categories(X_test):
   try:
      return X_test[0]
   except TypeError:
      return "None"
categorical_variables = ['Smoking_status' , 'Batch' , 'Current_tobacco_smoking' ,
                'Alcohol_intake_frequency'  , 'Time_since_last_menstrual_period' , 'Alcohol_drink_status'  ,'Sex'] 
for variable in categorical_variables:
   discarded = pd.get_dummies(X_test[variable],prefix = variable)        
   X_test= pd.concat([X_test, discarded], axis = 1)
   X_test.drop([variable], axis = 1, inplace = True) 


X_test

np.any(np.isnan(X_test))


### optimizing the parameters for ASXL1 ML algorithm --> gives a test AUC of 0.76 for ASLX1 (optimized parameters: criiterion = gini, max_depth = 7, 
### max_features= sqrt, min_samples_leaf = 8, n_estimators = 80

parameters = {'max_features':['sqrt'], 'n_estimators':[80,90,100,110,120,130],
              'max_depth':range(7,8,9),'min_samples_leaf':[2,3,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)

rf_class.fit(X=X_train, y=y_train)
rf_model = rf_class.best_estimator_

y_pred = rf_model.predict(X_test)

## Convert strings to ints to work with accuracy score functions
y_pred2 = pd.Series(y_pred).map({'No':0, 'Yes':1})
results = get_results(y_test, y_pred, results, 'Random Forest')
print('The best parameters are {}'.format(rf_class.best_params_))

features = X_train.columns
features

importances = rf_model.feature_importances_
importances

indices = np.argsort(importances)
indices

#### Plotting the Variable importance plot

plt.figure(figsize= (10,10))
features = X_train.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b',
         align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

result = permutation_importance(rf_model, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize = (10,10))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

plt.title('SHAP plot for CH')
shap.summary_plot(shap_values, features = X_test, class_inds = [1])
plt.show()

plt.title('SHAP Density plot for CH')
shap.summary_plot(shap_values[1], features = X_test)
plt.show()

## Accuracy

labels = ['No', 'Yes']
cm = confusion_matrix(y_test, y_pred, labels=labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Yes', 'No']) 
ax.yaxis.set_ticklabels(['Yes', 'No'])

plt.show()

results

print(results)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

train_probs = rf_model.predict_proba(X_train)[:,1] 

probs = rf_model.predict_proba(X_test)[:, 1]

train_predictions = rf_model.predict(X_train)
print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')
auc = metrics.roc_auc_score(y_test, probs)


probs

y_test

X_test


## Convert strings to ints to work with accuracy score functions
y_test2 = y_test.map({'No':0, 'Yes':1})
y_pred2 = pd.Series(y_pred).map({'No':0, 'Yes':1})

#define metrics
y_pred_proba =  y_pred2
fpr, tpr, _ = metrics.roc_curve(y_test2,  probs)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

print('Confusion Matrix : \n', cm)

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)
