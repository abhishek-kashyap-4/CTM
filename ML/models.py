

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:25:51 2023

@author: kashy
"""

from sklearn.model_selection import GridSearchCV, train_test_split ,learning_curve
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import GlobalVars 

#################################################################################################
def plot_learning_curve(model,X,y):
    
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
    )
    
    # Calculate mean and standard deviation of training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        color="blue",
        alpha=0.25,
    )
    plt.plot(train_sizes, test_mean, label="Cross-validation Score", color="green", marker="o")
    plt.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        color="green",
        alpha=0.25,
    )
    
    # Customize the plot
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
def get_test_report(model,Test,y,title,names=False,labels=GlobalVars.crop_list):
  predictions = model.predict(Test)
  if(names):
    predictions = [crops_inv[str(val)] for val in predictions]
  conf_matrix = confusion_matrix(y, predictions)
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', linewidths=0.5,xticklabels=labels,yticklabels=labels)
  plt.title(title)
  plt.show()
  
################################################################################################
#############################################      ML MODELS    ##########################################


def get_xgboost(df,y,tune = False , cv = False , size = 0.3,learning_curve=False,verbose=False):
    import xgboost as xgb
    num_classes = len(np.unique(y))
    df = df.values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=size)
    
    if(tune):
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
        grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        Parameters = grid_search.best_params_
        print("Best Parameters:", Parameters) if verbose else None 
    else:
        Parameters = {'colsample_bytree': 1.0, 'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1.0}
    if(cv):
        raise NotImplementedError 

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes,**Parameters)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(report) if verbose else None
    
    print('XGBOOST accuracy ',accuracy) 
    if(learning_curve):
        plot_learning_curve(model,df,y)
    return model,accuracy , report

def get_rf(df,y,tune = False , cv = False , size = 0.3,learning_curve=False,verbose=False):
    from sklearn.ensemble import RandomForestClassifier 
    if(cv):
        raise NotImplementedError 
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=size)
    
    rf = RandomForestClassifier()
    if(tune):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [ 'sqrt', 'log2'],}
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3)
        grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        params = grid_search.best_params_
        print("Best Parameters:", params)
    else:
        params = {}
        
    rf = RandomForestClassifier( **params)
    rf.fit(X_train,y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    conf_matrix = confusion_matrix(y_test, rf_predictions)
    #best_rf_params = best_rf_model.get_params()
    print("Random Forest Accuracy:", rf_accuracy)
    print("confusion matrix:",conf_matrix) if verbose else None 
    if(learning_curve):
        plot_learning_curve(rf,df,y)
    rf_report = classification_report(y_test,rf_predictions)
    print(rf_report) if verbose else None 
    
    return rf , rf_accuracy , rf_report 


def get_SVM(df,y,tune=False,cv=False , size = 0.3 , learning_curve = False , verbose = False ):
    from sklearn.svm import SVC
    if(cv):
        raise NotImplementedError 
    # Assuming you have a dataset with features (X) and target variable (y)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=size)

    if(tune):
      # Support Vector Machine (SVM) hyperparameter tuning
      svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
      svm_model = SVC()
      svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)
      svm_grid_search.fit(X_train, y_train)
      
      # Get the best parameters and build the SVM model
      params = svm_grid_search.best_params_
      print("Best params",params) if verbose else None 
    else:
      params = {}
    model = SVC(**params)
    model.fit(X_train,y_train)
    # Make predictions and evaluate the SVM model
    svm_predictions = model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    conf_matrix = confusion_matrix(y_test, svm_predictions)
    best_svm_params = model.get_params()
    print("SVM Accuracy:", svm_accuracy)
    if(learning_curve):
        plot_learning_curve(model,df,y)
    svm_report = classification_report(y_test,svm_predictions)
    print(svm_report) if verbose else None 
    return model , svm_accuracy , svm_report

'#########################################################################'




def model_wrapper(df,target,model = 'RF',tune='True',cv=False):
    if(model == 'RF'):
        
        model , accuracy , report = get_rf(df,target,tune = False , cv = False , size = 0.3,learning_curve=False,verbose=False) 
    elif(model == 'XGB'):
        
        model , accuracy , report = get_xgboost(df,target,tune = False , cv = False , size = 0.3,learning_curve=False,verbose=False) 
    elif(model == 'SVM'):
        
        model , accuracy , report  = get_SVM(df,target,tune=False,cv=False , size = 0.3 , learning_curve = False , verbose = False ) 
    elif(model == 'Catboost'):
        pass 
    elif(model == 'lstm'):
        pass 
    elif(model == 'cnn'):
        pass 
        
    else:
        raise Exception(f'Model {model} not recognized')
        
    return model ,accuracy , report




def pipeline_executable(first_arg,target = -1 , models = ['RF'] , tune = True , cv = False):
    if(not isinstance(target , (pd.Series,list,np.ndarray, np.generic))):
        raise Exception("Target column not prodived (correctly)")
    df = first_arg
    
    assert len(models)>0, "provide atleast 1 model option. Chose from list in config."
    d = {}
    best_accuracy = 0
    best_model = ''
    for modelname in models:
        model , accuracy , report = model_wrapper(df , target , model= modelname , tune = tune ,cv = cv)
        d[modelname] = (model , accuracy , report)
        if(accuracy > best_accuracy):
            best_accuracy = accuracy 
            best_model = modelname 
    print(f"Best accuracy is {best_accuracy} with {best_model}")
    return d

        
        
    
    


    

if __name__ == '__main__':
    pass 

