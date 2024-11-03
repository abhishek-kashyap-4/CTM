

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

def pipeline_executable(first_arg,model = 'RF' , tune = 'True' , cv = 10):

    if(model == 'RF'):
        pass 
    elif(model == 'XGB'):
        pass 
    elif(model == 'SVM'):
        pass 
    elif(model == 'Catboost'):
        pass 
    elif(model == 'lstm'):
        pass 
    elif(model == 'cnn'):
        pass 
        
    else:
        raise Exception(f'Model {model} not recognized')
    

if __name__ == '__main__':
    pass 

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.optimizers import Adam

from global_vars import cropslist,crops_inv


from sklearn.model_selection import 
from keras.layers import Conv1D, MaxPooling1D, Flatten



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



'########################################################################'
'Tuning, plotting and learning curve. '
'1. Random Forest'
'2. XGBoost'
'3. SVM'
import xgboost as xgb
def get_xgboost(df,y,learning_curve=False):
    num_classes = len(np.unique(y))
    df = df.values
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
        }
    Parameters = {'colsample_bytree': 1.0, 'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1.0}
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes,**Parameters)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    print(classification_report(y_test, y_pred))
    print('XGBOOST accuracy ',accuracy_score(y_test, y_pred))
    if(learning_curve):
        plot_learning_curve(model,df,y)
    return model,classification_report(y_test, y_pred)
def get_xgboost_grid(df,y):
    num_classes = len(np.unique(y))
    df = df.values
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
    # Define a parameter grid for hyperparameter tuning
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    best_model = xgb.XGBClassifier(objective='multi:softmax',num_class=num_classes,
        **best_params )

    best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = best_model.predict(X_test.reshape(X_test.shape[0], -1))
    print(classification_report(y_test, y_pred))
    
def get_rf(df,y,learning_curve=False):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    # Make predictions and evaluate the Random Forest model
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    conf_matrix = confusion_matrix(y_test, rf_predictions)
    #best_rf_params = best_rf_model.get_params()
    print("Random Forest Accuracy:", rf_accuracy)
    print("confusion matrix:",conf_matrix)
    if(learning_curve):
        plot_learning_curve(rf,df,y)
    rf_report = classification_report(y_test,rf_predictions)
    print(rf_report)
    return rf

def get_rf_grid(df,y):
    df = df.values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
    model = RandomForestClassifier()# Define a parameter grid for hyperparameter tuning
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [ 'sqrt', 'log2'],}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Print the best parameters found by GridSearchCV
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train the model with the best parameters
    best_model = RandomForestClassifier( **best_params)
    best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Print the classification report
    print(classification_report(y_test, y_pred))
'#########################################################################'

def SVM_RF(df,y,ml_row,tune=False,plot=False,dosvm=True,dorf=True):
  # Assuming you have a dataset with features (X) and target variable (y)
  X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4)
  assert dosvm==True or dorf==True
  best_svm_model , best_rf_model = -1,-1
  if(dosvm):
      if(tune):
        # Support Vector Machine (SVM) hyperparameter tuning
        svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
        svm_model = SVC()
        svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)
        svm_grid_search.fit(X_train, y_train)
    
        # Get the best parameters and build the SVM model
        best_svm_params = svm_grid_search.best_params_
        best_svm_model = SVC(**best_svm_params)
        best_svm_model.fit(X_train, y_train)
    
         # Make predictions and evaluate the SVM model
        svm_predictions = best_svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        conf_matrix = confusion_matrix(y_test, svm_predictions)
        print("SVM Accuracy:", svm_accuracy)
        print()
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
      else:
        best_svm_model = SVC()
        best_svm_model.fit(X_train,y_train)
        # Make predictions and evaluate the SVM model
        svm_predictions = best_svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        conf_matrix = confusion_matrix(y_test, svm_predictions)
        best_svm_params = best_svm_model.get_params()
        print("SVM Accuracy:", svm_accuracy)
        print()
        if(plot):
          plt.figure(figsize=(10, 8))
          sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
          plt.title('SVM Confusion Matrix')
          plt.show()
        svm_report = classification_report(y_test,svm_predictions)
        print(svm_report)
  if(dorf):
      if(tune):
        # Random Forest hyperparameter tuning
        rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]}
        rf_model = RandomForestClassifier()
        rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5)
        rf_grid_search.fit(X_train, y_train)
    
        # Get the best parameters and build the Random Forest model
        best_rf_params = rf_grid_search.best_params_
        best_rf_model = RandomForestClassifier(**best_rf_params)
        best_rf_model.fit(X_train, y_train)
    
        # Make predictions and evaluate the Random Forest model
        rf_predictions = best_rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        conf_matrix = confusion_matrix(y_test, rf_predictions)
        print("Random Forest Accuracy:", rf_accuracy)
        rf_report = classification_report(y_test,rf_predictions)
        print(rf_report)
      else:
        best_rf_model = RandomForestClassifier()
        best_rf_model.fit(X_train,y_train)
        # Make predictions and evaluate the Random Forest model
        rf_predictions = best_rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        conf_matrix = confusion_matrix(y_test, rf_predictions)
        best_rf_params = best_rf_model.get_params()
        print("Random Forest Accuracy:", rf_accuracy)
        if(plot):
          plt.figure(figsize=(10, 8))
          sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
          plt.title('RF Confusion Matrix')
          plt.show()
    
        rf_report = classification_report(y_test,rf_predictions)
        print(rf_report)
  if(dosvm):
      ml_row['SVM_Parameters'] = best_svm_params
      ml_row['SVM_accuracy'] = svm_accuracy
      ml_row['SVM_Report'] = svm_report
  if(dorf):
      ml_row['RF_Parameters'] = best_rf_params
      ml_row['RF_Accuracy'] = rf_accuracy
      ml_row['RF_Report'] = rf_report


  return ml_row , best_svm_model,best_rf_model

def get_test_report(model,Test,y,title,names=False,labels=cropslist):
  predictions = model.predict(Test)
  if(names):
    predictions = [crops_inv[str(val)] for val in predictions]
  conf_matrix = confusion_matrix(y, predictions)
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', linewidths=0.5,xticklabels=labels,yticklabels=labels)
  plt.title(title)
  plt.show()
  
def fit_lstm(X,y,epochs = 20,batch_size=25):
    timesteps = X.shape[1]
    X = X.reshape(X.shape[0], timesteps, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    model = Sequential()
    model.add(LSTM(50, input_shape=(timesteps, 1)))  # Adjust the number of LSTM units based on your problem
    model.add(Dense(units=3, activation='softmax'))  # Adjust num_classes based on your problem
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

from keras.utils import to_categorical

def multivariate_lstm(X_toshape,y,columns = ['NDVI','NDYI','VV_VH'],epochs=20, batch_size=35,lr=0.02,timesteps = [0,1,2,3,4,5,6,7,8,9,10]):
    
    '''
    X is 2d With <samples, time and features>
    it needs to be <samples, time, features>
    
    such that, X[i,:,0] gives the time series of a feature
    
    '''    
    glob = 0
    X = np.ndarray(shape=(X_toshape.shape[0],len(timesteps),len(columns)))
    for co in columns:
        select = [str(i)+'_'+co for i in timesteps]
        X[:,:,glob] = X_toshape[select]
        glob+=1
    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes=num_classes)
    #y_categorical =np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
    model = Sequential()
    model.add(LSTM(units=64,return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=32))
    model.add(Dense(units=num_classes, activation='softmax'))
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test) ) #,verbose=0)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    y_test_actual = np.argmax(y_test,axis=1)
    print(classification_report(y_test_actual, y_pred))
    

    #plot_learning_curve(model, X, y_categorical)
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    return model,accuracy,classification_report(y_test_actual, y_pred)

def CNN_LSTM(X_toshape,y,columns = ['NDVI','NDYI','VV_VH'],epochs=50, batch_size=5,lr=0.02,timesteps = [0,1,2,3,4,5,6,7,8,9,10]):
    glob = 0
    X = np.ndarray(shape=(X_toshape.shape[0],len(timesteps),len(columns)))
    for co in columns:
        select = [str(i)+'_'+co for i in timesteps]
        X[:,:,glob] = X_toshape[select]
        glob+=1
    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
    
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=50))
    model.add(Dense(units=num_classes, activation='softmax'))
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    #plot_learning_curve(model, X, y_categorical)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    y_test_actual = np.argmax(y_test,axis=1)
    print(classification_report(y_test_actual, y_pred))
    

    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    return model,accuracy
'######################DEEPER###################'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(train_loader, input_size, hidden_size, output_size, num_layers, num_epochs, learning_rate):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Example usage:
# train_loader should be a DataLoader with your training data
# input_size, hidden_size, output_size, num_layers are hyperparameters
# num_epochs and learning_rate are training parameters

# train_lstm_model(train_loader, input_size, hidden_size, output_size, num_layers, num_epochs, learning_rate)
    
'####################################################################################'

class ConvNetModel(nn.Module):
    def __init__(self, input_channels, output_size,input_size):
        super(ConvNetModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (input_size // 2), output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * (x.size // 2))
        x = self.fc1(x)
        return x

def train_convnet_model(train_loader, input_channels, output_size,input_size, num_epochs, learning_rate):
    model = ConvNetModel(input_channels, output_size,input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Example usage:
# train_loader should be a DataLoader with your training data
# input_channels, output_size are hyperparameters
# num_epochs and learning_rate are training parameters

# train_convnet_model(train_loader, input_channels, output_size, num_epochs, learning_rate)
'#######################################################################'


def prepare_data_for_lstm_convnet(features, labels, sequence_length, batch_size, shuffle=True):
    """
    Prepare time series data for training LSTM and ConvNet models.

    Args:
        features (numpy.ndarray): Input features as a 2D array (num_samples x num_features).
        labels (numpy.ndarray): Labels as a 1D array.
        sequence_length (int): Length of sequences for LSTM.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: DataLoader for training.
    """
    num_samples, num_features = features.shape
    num_classes = len(set(labels))

    # Reshape data for LSTM
    lstm_input = torch.tensor(features, dtype=torch.float32).view(num_samples, sequence_length, num_features)
    lstm_labels = torch.tensor(labels, dtype=torch.long)

    # Reshape data for ConvNet
    convnet_input = torch.tensor(features, dtype=torch.float32).view(num_samples, 1, num_features)
    convnet_labels = torch.tensor(labels, dtype=torch.long)

    # Create TensorDataset
    lstm_dataset = TensorDataset(lstm_input, lstm_labels)
    convnet_dataset = TensorDataset(convnet_input, convnet_labels)

    # Create DataLoaders
    lstm_loader = DataLoader(lstm_dataset, batch_size=batch_size, shuffle=shuffle)
    convnet_loader = DataLoader(convnet_dataset, batch_size=batch_size, shuffle=shuffle)

    return lstm_loader, convnet_loader

# Example usage:
# lstm_loader, convnet_loader = prepare_data_for_lstm_convnet(features, labels, sequence_length, batch_size)


'''
# Assuming you have your features and labels ready
# Adjust hyperparameters accordingly

sequence_length = 10
batch_size = 32
input_channels = 1
output_size = 5  # Number of classes
num_epochs = 10
learning_rate = 0.001

# Step 1: Prepare Data
lstm_loader, convnet_loader = prepare_data_for_lstm_convnet(features, labels, sequence_length, batch_size)

# Step 2: Train LSTM Model
lstm_model = train_lstm_model(lstm_loader, input_size, hidden_size, output_size, num_layers, num_epochs, learning_rate)

# Step 3: Train ConvNet Model
convnet_model = train_convnet_model(convnet_loader, input_channels, output_size, input_size, num_epochs, learning_rate)

'''