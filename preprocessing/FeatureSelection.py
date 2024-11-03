# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:17:32 2024

@author: kashy
"""
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif

import pandas as pd

def IFE2(data,target,by='var',correlation_threshold=0.8,cutoff=15,verbose=False):

    selected_features = []
    datacorr = data.corr()
    #removed = []
    while data.shape[1] > 0:
        cutoff -= 1 
        if(cutoff<1):
            break
        if(by=='var'):
          feature_variances = data.var()
          selected_feature = feature_variances.idxmax()
        elif(by=='covar'):
            selected_feature = data.corrwith(target).abs().sort_values(ascending=False).index[0]
    
        selected_features.append(selected_feature)
        print(selected_feature ,"Selected") if verbose else -1

        correlated_features = data.corr()[selected_feature].abs()

        correlated_features = correlated_features[correlated_features > correlation_threshold].index
        #correlated_features = [feature for feature in correlated_features if feature not in removed]
        #print("Correlated Features are ",correlated_features) if verbose else -1
        data = data.drop(correlated_features, axis=1)
        #removed += correlated_features
    
    return selected_features


def IFE(data,target,by='var',model=None,correlation_threshold=0.4,verbose=False,plot=False,waitforinput=False):
  '''
  Iterative Feature Selection.
  by = model, var, covar 
  IFE(by='var',correlation_threshold=0.4,verbose=True,plot=True,waitforinput=True)
  '''


  selected_features = []
  print('Method:' ,by) if verbose==True else -1
  while data.shape[1] > 0:
    input() if waitforinput else -1
    if(by=='var'):
      feature_variances = data.var()
      selected_feature = feature_variances.idxmax()
    elif(by=='model'):
        if(model==None):
            model= RAFO(data,target)
        else:
            model.fit(data,target)
        selected_feature = data.columns[model.feature_importances_.argmax()]
    elif(by=='covar'):
        selected_feature = data.corrwith(target).abs().sort_values(ascending=False).index[0]
    elif(by=='anova'):
        #F-Test (ANOVA)
        f_test_selector = SelectKBest(score_func=f_classif, k=1)
        selected_feature = data.columns[f_test_selector.fit(data, target).get_support()].tolist()[0]
    elif(by=='mi'):
        #Mutual Information
        mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=1)
        selected_feature = data.columns[mutual_info_selector.fit(data, target).get_support()].tolist()[0]
    else:
      raise Exception(f'Unrecognized value for by parameter, {by}')

    selected_features.append(selected_feature)
    print(selected_feature ,"Selected") if verbose else -1

    correlated_features = data.corr()[selected_feature].abs()
    plt.plot(correlated_features) if plot else -1
    plt.show() if plot else -1
    correlated_features = correlated_features[correlated_features > correlation_threshold].index

    print("Correlated Features are ",correlated_features) if verbose else -1
    data = data.drop(correlated_features, axis=1)
  return selected_features

def Copt(by='accuracy',model=-1):
  '''
  Combination optimization
  Take all combinations of columns and fit randomforest
  '''
  if(model==-1):
    raise Exception("You need a model to run Copt.")

def pipeline_executable(first_arg , target=-1 , by='covar',correlation_threshold=0.7,cutoff=15,verbose=True):
    
    if(not isinstance(target, (pd.DataFrame,pd.Series, list, dict))) :
        raise Exception("Target wasn't specified")
    df = first_arg 
    selected_features = IFE2(df , target=target ,by=by,correlation_threshold=correlation_threshold,cutoff = 15,verbose=verbose) 
    return df[selected_features]
    
if __name__=='__main__':
    df = pd.read_csv('Data/interim/satellite/all_optical_10day_fixedJan1_featuredAdded.csv')
    
    target = df['Crop_Type']
    df = df.drop(columns = ['Crop_Type']) 
    
    # A_K_ this is error prone. Come back to this.
    print("WARNING - removing hm features manually for feature selection. (and not adding back). This is error-prone,change it.")
    cols = ['Unique_Id'  ]
    df = df.drop(columns = cols) 
    
    if(df.isnull().values.any()):
        1/0
        
    df = pipeline_executable(df,target,by='covar',correlated_threshold=0.7,verbose=True)
    
    if(df.isnull().values.any()):
        1/0
    
    addeddf = pipeline_executable(df)
    addeddf.to_csv('Data/interim/satellite/all_optical_10day_fixedJan1_featuredAddedSelected.csv')
    
    
