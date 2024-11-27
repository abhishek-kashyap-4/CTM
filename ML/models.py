

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

from utils import utils 


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


from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input , Concatenate
from keras.optimizers import Adam

import re 

def reshape_to_3d(X_toshape):
    '''
    X is 2d With <samples, time and features>
    it needs to be <samples, time, features>
    
    such that, X[i,:,0] gives the time series of a feature
    
    '''  
    timesteps , columns = utils.get_timesteps_bands(X_toshape , check=True)
    
    glob = 0
    X = np.ndarray(shape=(X_toshape.shape[0],len(timesteps),len(columns)))
    for co in columns:
        select = [str(i)+'__'+co for i in timesteps]
        X[:,:,glob] = X_toshape[select]
        glob+=1
    return X 
    
def get_LSTM(df,y ):
    '''
    df is expected to have 
        - Time series columns
        - Aggregated Columns 
        
    Feature Selected is expect to be done before this.  
    '''
    params = {
              'epochs': 20 , 
              'batch_size': 35, 
              'lr': 0.02 }
    tcols = []
    ocols = []
    for col  in df.columns: 
        if  re.match(r'[0-9]',col):
            tcols.append(col)
        else:
            ocols.append(col)
            
        
    X_toshape = df[tcols]
    X_time = reshape_to_3d(X_toshape)
    
    X_others = df[ocols]
    
    num_classes = len(np.unique(y))
    # I think it's already happenning
    
    #y_categorical = to_categorical(y, num_classes=num_classes)
    
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(indices, test_size=0.3)
    Xtime_train, Xtime_test = X_time[train_indices], X_time[test_indices]
    Xothers_train, Xothers_test = X_others[train_indices], X_others[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    

    time_series_input = Input(shape=(Xtime_train.shape[1], Xtime_train.shape[2]))  # e.g., (20, 2) for 20 steps, 2 features
    lstm_1 = LSTM(64, return_sequences=True)(time_series_input)  # 64 units
    lstm_output = LSTM(32, return_sequences=False)(lstm_1)
    

    standalone_input = Input(shape=(Xothers_train.shape[1],))  # e.g., (1,) for MaxNDVI
    dense_features = Dense(32, activation='relu')(standalone_input)
    
    combined = Concatenate()([lstm_output, dense_features])
    final_output = Dense(num_classes, activation='softmax')(combined)  

    from tensorflow.keras.models import Model

    model = Model(inputs=[time_series_input, standalone_input], outputs=final_output)
    adam = Adam(learning_rate=params['lr'])
    
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = model.fit([Xtime_train,Xothers_train], y_train,
                       epochs=params['epochs'], batch_size=params['batch_size'], 
                       shuffle=True,validation_data=([Xtime_test,Xothers_test], y_test) ,
                       callbacks = [early_stopping , reduce_lr]) 

    
    import matplotlib.pyplot as plt
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return model , history.history['accuracy'], history
    
    




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
        
        model , accuracy , report  = get_LSTM(df,target) 
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

