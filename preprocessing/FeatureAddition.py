# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:13:07 2024

@author: kashy
"""
import pandas as pd 
import re
import warnings


from utils.utils import get_timesteps_bands , check_column_syntax 

def feature_addition(df, time_indexes = [], comb=True , tim = False , STN = False, tim_method = 'Full', STN_w = 3):
    '''
    Add features as columns to the dataframe. 
    Features are of the following types:
        1. Single timestep band combinations - Ex. NDVI
        2. Multiple timestep combinations (single band, usually) - Ex. maxNDVI.
            2.1 'Full' : Add paper 1 implementation.
            2.2 'Once' : Add just one index for all timesteps
            2.3 'All_Agg' : Aggregate a time index for every timestep.
        3. STN - Sliding temporal window 
    NOTES:
        -> compute timesteps since different versions of df can be passed.
        -> Do not Impose YYYYMMDD. Since gdd based can be sent. Thank you for this comment, AK.
        -> This means regex is always [0-9]+__ 
    '''
      
    indices = []
    timesteps,indices = get_timesteps_bands(df,check=True)

    dfd = {}
    
    assert comb or tim or STN , "No feature addition is selected from Comb, Tim, STN. Why are you calling this function?" 
    
    ########### 1. Combination Indexes.
    if(comb):
        for time in timesteps:
          dfd[str(time)+'__NDVI'] = (df[str(time)+'__B8'] - df[str(time)+'__B4'])/(df[str(time)+'__B8'] + df[str(time)+'__B4'])
          L= 0.5
          dfd[str(time)+'__SAVI'] = ((df[str(time)+'__B8'] - df[str(time)+'__B4']) / (df[str(time)+'__B8'] + df[str(time)+'__B4'] + L)) * (1 + L)
          dfd[str(time)+'__NDRE'] = (df[str(time)+'__B7'] - df[str(time)+'__B5']) / (df[str(time)+'__B7'] + df[str(time)+'__B5'])
          dfd[str(time)+'__NDWI'] =  (df[str(time)+'__B3'] - df[str(time)+'__B8']) / (df[str(time)+'__B3'] + df[str(time)+'__B8'])
          dfd[str(time)+'__NDYI'] = (df[str(time)+'__B3'] - df[str(time)+'__B2'])/(df[str(time)+'__B3'] + df[str(time)+'__B2'])
      
          #df[str(time)+ '_LAI'] = (1 / (-0.35)) * np.log((0.69 - df[str(time)+'_B4']) / (0.69 + df[str(time)+'_B4']))
          # LAI creating null values
        #Be careful, this needs to be manually updated when you add a new index above.
        df = pd.concat([df,pd.DataFrame(dfd)],axis=1)
        indices += ['NDVI','SAVI','NDRE','NDWI','NDYI'] 
    ########### 2. Time Aggregated Indexes.
    if(tim):
        # Adding time indexes.
        d = {}
        band_columns = lambda band: [col for col in df.columns if(re.match(s+r'[0-9]+__'+band,col))]
        band_max = lambda band :df[band_columns(band)].max(axis=1)
        band_min = lambda band: df[band_columns(band)].min(axis=1)
        band_mean = lambda band: df[band_columns(band)].mean(axis=1)
        band_max_gradient = lambda band: df[band_columns(band)].diff(axis=1).drop(f'{ft}__'+band, axis=1).max(axis=1)
      
        #######################
        """
        Paper 1:
        NDVI_max , t_max
        NDVI_inf_1 , t_inf_1 #Max Gradient value, time
        NDVI_inf_2 , t_inf_2 #min Gradient Value, time
        Delta_NDVI , FGP - NDVI_max-NDVI_inf_1 , t_max-t_inf_1
        """
        def band_params(band):
           # Remember, regex is NOT YYYYMMDD__band, since gdd scaled can be passed.
           #The regex is numbers of any length followed by double underscore.
          cols = [col for col in df.columns if(re.match(r'[0-9]+__'+band,col))]
          subdf = df[cols]
          max_values = []
          t_max = []
      
          maxGradient = []
          inf_1 = []
          t_inf_1 = []
      
          minGradient = []
          inf_2 = []
          t_inf_2 = []
      
          mean_values = []
          delta = []
          fgp = []
      
          for rownum,row in subdf.iterrows():
            max_values.append(row.max())
            t_max.append(row.argmax())
            nr = row.diff().drop('0__'+band)
      
            maxGradient.append(nr.max())
            inf_1.append(row.iloc[nr.argmax()+1]) #+1 Because we drop 0
            t_inf_1.append(nr.argmax())
      
            minGradient.append(nr.min())
            inf_2.append(row.iloc[nr.argmin()+1])
            t_inf_2.append(nr.argmin())
      
            mean_values.append(row.mean())
            delta.append(row.max() - row.iloc[nr.argmax()+1])
            fgp.append(row.argmax()-nr.argmax())
      
      
          d = {}
          d['max_'+index] = max_values
          d['t_max_'+index] = t_max
          d['maxGradient_'+index] = maxGradient
          d['inf_1_'+index] = inf_1
          d['t_inf_1_'+index] = t_inf_1
          d['minGradient_'+index] = minGradient
          d['inf_2_'+index] = inf_2
          d['t_inf_2_'+index] = t_inf_2
          d['mean_'+index] = mean_values
          d['delta_'+index] = delta
          d['fgp_'+index] = fgp
          return d
        #########################
        if(tim_method=='Full'):
          d = {}
          for index in indices:
            smol_d = band_params(index)
            d.update(smol_d)
      
        elif(tim_method=='All_Agg'):
          print("Sorry, All_Agg hasn't been implemented yet. I am working on it tho.")
        elif(tim_method=='Once'):
      
          ft , tt = 0,9 # Change these parameters if you want time indexes between different timesteps.
          
          s = fr'[{ft}-{tt}]'  # f for format , inserting ft and tt; r for raw string, for regex.
          for index in indices:
            d['max'+index] = band_max(index)
            d['min'+index] = band_min(index)
            d['maxGradient'+index] = band_max_gradient(index)
            d['mean'+index] = band_mean(index)
        else:
          raise Exception("method {} for time indexes isn't recognized. Use 'Full','All_Agg', or 'Once' (or code your own...) ".format(tim_method))
      
      
      
        if(time_indexes == 'all'):
          df = pd.concat([df, pd.DataFrame(d)], axis=1)
        else:
          for key in time_indexes:
            if(key in d.keys()):
              df[key] = d[key]
      
       
    ########### 3. Sliding Temporal Window.
    if(STN):
      raise Exception('Though STN is implemented, I did not check it yet. Use this option once I verify it.')
        
      w = STN_w
      sd = {}
      for time in timesteps[w:-1]:
        ft = time-w # this is being used by the lambda functions
        s = fr'[{time-w}-{time}]'
        band_columns = lambda band: [col for col in df.columns if(re.match(s+r'_'+band,col))]
        for index in indices:
          sd[f'{w}stn_'+str(time)+'_max'+index] = band_max(index)
          sd[f'{w}stn_'+str(time)+'_min'+index] = band_min(index)
          sd[f'{w}stn_'+str(time)+'_maxGradient'+index] = band_max_gradient(index)
          sd[f'{w}stn_'+str(time)+'_mean'+index] = band_mean(index)
      d.update(sd)
      
      # A_K_ adding this. 
      df = pd.concat([df, pd.DataFrame(d)], axis=1)
      #for key in d:
       #   df[key] = d[key]
          
      

    if(df.isnull().values.any()):
        warnings.warn("DataFrame contains null values. Could be the creation of indices.",UserWarning)

        
    return df
  
def additional(df):
    '''
    Use this script to add additional columns. 
    1. Geography - Ag zone. 
    
    '''
    # A_K_ come back to this. 
    
def pipeline_executable(first_arg , time_indexes = 'all', comb=True , tim = False , STN = False, tim_method = 'Full', STN_w = 3):
    df = first_arg
    
    assert sum(1 for col in df.columns if '__' in col) >1 , "Follow the convention of <Date/timestep>__<band> (Double Underscore)"
    addeddf = feature_addition(df,time_indexes = time_indexes,comb=comb,tim=tim,STN=STN , tim_method = tim_method,STN_w=STN_w)
    return addeddf


if __name__ == '__main__':
    
    df = pd.read_csv('Data/Interim/Preprocessed/FieldOptical_CGDD.csv')
    check_column_syntax(df , kind = 'timestep')
    
    warnings.warn("Unique_Id has null values. I think its from sjoin in EnRtext_to_csv2. you've to correct that.",UserWarning)
    df = df.dropna(subset = ['Unique_Id'])

    
    addeddf = pipeline_executable(df)
    addeddf.to_csv('Data/Interim/Added/FieldOptical_CGDD.csv')
    
    

