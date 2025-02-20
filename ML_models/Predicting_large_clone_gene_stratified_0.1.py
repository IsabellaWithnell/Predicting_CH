import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, roc_curve, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


#  compute performance metrics
def get_results(truth, prediction):
    # Map string responses to numeric values
    y_true = truth.map({'No': 0, 'Yes': 1})
    y_pred = pd.Series(prediction).map({'No': 0, 'Yes': 1})
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }


# Function to run the Random Forest model on a given DataFrame for one gene
def run_rf_model(data, gene_label):
    print(f"\n--- Processing {gene_label} ---")
    
    # Print and set up the response variable (assumed to be the last column)
    print("Response variable counts (before mapping):")
    print(data.iloc[:, -1].value_counts())
    data.iloc[:, -1] = data.iloc[:, -1].map({'No': 'No', 'Yes': 'Yes'})
    print("Response variable counts (after mapping):")
    print(data.iloc[:, -1].value_counts())
    data.iloc[:, -1] = data.iloc[:, -1].astype('category')
    
    # Convert specified columns to category
    cat_cols = ['Sex', 'Batch', 'Smoking_status', 'Alcohol_intake_frequency',
                'Time_since_last_menstrual_period', 'Alcohol_drink_status', 'Current_tobacco_smoking']
    for col in cat_cols:
        data[col] = data[col].astype('category')
    
    # Split data: features are all columns except the last; response is the last column
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    
    # One-hot encode the categorical variables in training and test sets
    for col in cat_cols:
        X_train = pd.concat([X_train, pd.get_dummies(X_train[col], prefix=col)], axis=1)
        X_train.drop(columns=[col], inplace=True)
        X_test = pd.concat([X_test, pd.get_dummies(X_test[col], prefix=col)], axis=1)
        X_test.drop(columns=[col], inplace=True)
    
    # Define hyperparameter grid
    param_grid = {
        'max_features': ['sqrt'],
        'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130],
        'max_depth': range(2, 12),
        'min_samples_leaf': [2, 3, 4, 6, 8],
        'criterion': ['gini', 'entropy']
    }
    
    # Perform grid search with a RandomForestClassifier
    grid_search = GridSearchCV(RandomForestClassifier(random_state=8), param_grid, n_jobs=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predict on test set and compute performance metrics
    y_pred = best_model.predict(X_test)
    metrics_dict = get_results(y_test, y_pred)
    
    print("Best parameters:", grid_search.best_params_)
    print("Performance metrics:", metrics_dict)
    
    # Compute ROC AUC scores on train and test sets
    train_probs = best_model.predict_proba(X_train)[:, 1]
    test_probs = best_model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train.map({'No': 0, 'Yes': 1}), train_probs)
    test_auc = roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), test_probs)
    print(f"Train ROC AUC Score: {train_auc}")
    print(f"Test ROC AUC Score: {test_auc}")
    
    # Plot ROC curve for test set
    y_test_numeric = y_test.map({'No': 0, 'Yes': 1})
    fpr, tpr, _ = roc_curve(y_test_numeric, test_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    plt.title(f"ROC Curve for {gene_label}")
    plt.show()
    
    # Calculate confusion matrix and derived metrics
    labels = ['No', 'Yes']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    total = cm.sum()
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("Confusion Matrix:\n", cm)
    print("Calculated Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    
    # Return a dictionary of results for the gene
    return {
        'gene': gene_label,
        'best_params': grid_search.best_params_,
        'metrics': metrics_dict,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


# Dictionary of file paths for each gene 
file_paths = {
    'DNMT3A': '/rds/general/user/iw413/home/Summerproject/outputs/JanDNMT3Alargeclone01.csv',
    'ASXL1': '/rds/general/user/iw413/home/Summerproject/outputs/JanASXL1largeclone01.csv',
    'ATM': '/rds/general/user/iw413/home/Summerproject/outputs/JanATMlargeclone01.csv',
    'CALR': '/rds/general/user/iw413/home/Summerproject/outputs/JanCALRlargeclone01.csv',
    'CBL': '/rds/general/user/iw413/home/Summerproject/outputs/JanCBLlargeclone01.csv',
    'GNAS': '/rds/general/user/iw413/home/Summerproject/outputs/JanGNASlargeclone01.csv',
    'GNB1': '/rds/general/user/iw413/home/Summerproject/outputs/JanGNB1largeclone01.csv',
    'JAK2': '/rds/general/user/iw413/home/Summerproject/outputs/JanJAK2largeclone01.csv',
    'KMT2A': '/rds/general/user/iw413/home/Summerproject/outputs/JanKMT2Alargeclone01.csv',
    'KRAS': '/rds/general/user/iw413/home/Summerproject/outputs/JanKRASlargeclone01.csv',
    'NF1': '/rds/general/user/iw413/home/Summerproject/outputs/JanNF1largeclone01.csv',
    'PPM1D': '/rds/general/user/iw413/home/Summerproject/outputs/JanPPM1Dlargeclone01.csv',
    'SF3B1': '/rds/general/user/iw413/home/Summerproject/outputs/JanSF3B1largeclone01.csv',
    'SRCAP': '/rds/general/user/iw413/home/Summerproject/outputs/JanSRCAPlargeclone01.csv',
    'SRSF2': '/rds/general/user/iw413/home/Summerproject/outputs/JanSRSF2largeclone01.csv',
    'STAT3': '/rds/general/user/iw413/home/Summerproject/outputs/JanSTAT3largeclone01.csv',
    'TET2': '/rds/general/user/iw413/home/Summerproject/outputs/JanTET2largeclone01.csv',
    'TP53': '/rds/general/user/iw413/home/Summerproject/outputs/JanTP53largeclone01.csv',
    'TTN': '/rds/general/user/iw413/home/Summerproject/outputs/JanTTNlargeclone01.csv',
    'YLPM1': '/rds/general/user/iw413/home/Summerproject/outputs/JanYLPM1largeclone01.csv',
    'ZBTB33': '/rds/general/user/iw413/home/Summerproject/outputs/JanZBTB33largeclone01.csv',
    'ZNF318': '/rds/general/user/iw413/home/Summerproject/outputs/JanZNF318largeclone01.csv'
}

# Run the Random Forest model for each gene and store results in a dictionary
results_dict = {}
for gene, path in file_paths.items():
    df = pd.read_csv(path)
    results_dict[gene] = run_rf_model(df, gene)

print("\n--- Summary of Test AUC and Accuracy for each gene ---")
for gene, res in results_dict.items():
    print(f"{gene}: Test AUC = {res['test_auc']:.3f}, Accuracy = {res['accuracy']:.3f}")
