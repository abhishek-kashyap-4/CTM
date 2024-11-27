# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:14:19 2024

@author: kashy
"""
import re
import numpy as np
import pandas as pd
import warnings 

from utils import utils 
from tqdm import tqdm

# A_K_ come back to this. 

'''
All the preprocessing steps you have to do
1. impose YYYY_MM_DD__<bandname>
2. Check for null values 
3. Remap target column
4. Check  if all columns have the same timesteps 
5. Normalize 
6. 

'''

def kalman_fill(df, columns):
    from pykalman import KalmanFilter
    """
    Applies Kalman filtering to fill missing values in specified columns of a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing time series data.
    - columns (list of str): List of column names to apply Kalman filtering.
    
    Returns:
    - pd.DataFrame: A copy of the DataFrame with missing values filled in the specified columns.
    """
    df_filled = df.copy()

    for column in columns:
        series = df[column].values
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        series_filled_temp = np.nan_to_num(series, nan=np.nanmean(series))
        kf = kf.em(series_filled_temp, n_iter=5)
        smoothed_state_means, _ = kf.smooth(series)
        df_filled[column] = np.where(np.isnan(series), smoothed_state_means.flatten(), series)

    return df_filled



def null_method(df , method ):
    ' Method nulls = dont , remrow , remcol , strategy1 ,interpol_pass'
    if(method == 'dont'):
        pass 
    elif(method == 'remrow'): 
        df = utils.filter_df_nulls(df,method = method , perc = 0 ) 
    elif(method =='remcol'):
        df = utils.filter_df_nulls(df,method = method , perc = 0 ) 
    elif( re.match('strategy',method)):
        assert method.split('_')[-1] =='all' , f" Method {method} requesting all rows not be removed. Currently, This isn't possible."
        perc = int(method.split('_')[1]) 
        assert perc >=0 and perc <=100 , f"For method {method}, the number has to be in 0 -100"
        perc /=100  
        df = utils.filter_df_nulls(df,method = 'remcol' , perc = perc)
        df = utils.filter_df_nulls(df,method = 'remrow' , perc = 0)

    elif(re.match('interpol',method)):
        method = method.split('_')[-1]
        
        if(method == 'pass'):
            timesteps , bands = utils.get_timesteps_bands(df,reg = '[0-9]+__')
            for band in tqdm(bands):
                cols = [str(time)+'__'+ band for time in timesteps]
                df[cols] = df[cols].ffill(axis = 1).bfill(axis=1)
        elif(method == 'kelman'):
            timesteps , bands = utils.get_timesteps_bands(df,reg='[0-9]+__')
            # Correct, but bad way of assigning df 
            for band in tqdm(bands):
                cols = [str(time)+'__'+ band for time in timesteps]
                df = kalman_fill(df,cols)
            
        else:
            raise ValueError(f" Interpol method {method} not recognized. Look at config file for options.")
    
    else:
        raise ValueError(f" method {method} not recognized. Look at config file for options.")
    
    return df 
    
def smoothing(df , method = 'sav_gol',window = 5 , polyorder = 2):
    '''
    Apply a smoothing technique on the dataframe.
    '''
    
    if(method == 'sav_gol'):
        from scipy.signal import savgol_filter
        timesteps , bands = utils.get_timesteps_bands(df,reg = r'[0-9]+__') 
        for band in bands:
            cols = [str(time)+'__'+band for time in timesteps]
            for rownum, row in df[cols].iterrows():
                savgol_filter(row, window, polyorder)
            
            df[cols] = pd.DataFrame(
                    df[cols].apply(lambda x: savgol_filter(x.values, window, polyorder), axis=1).tolist(),
                    index=df.index,
                    columns=cols
                )
            #df[cols]  = df[cols].apply(lambda x: savgol_filter(x.values, window, polyorder), axis=1) 
    else:
        raise Exception("Any other method than savgol isn't recognized")
    
    return df 
    

# COnsider using **kwargs
def preprocess(df,targetname = 'CropType' , which = 'Optical' ,mapper = {}, impose_date=True,method_nulls = 'interpol_pass' ,method_remap = 'Lazy' , method_smooth= 'sav_gol',window=5,polyorder = 2):
    
    
    reg = r'[0-9]{8}__' if impose_date else r'[0-9]+__'
    utils.check_column_syntax(df,kind = 'custom' , reg = reg )
    
    
    df = null_method(df , method = method_nulls)
    null_positions = utils.null_positions(df)
    
    #assert len(null_positions) == 0 ,f"{len(null_positions)}"
    ## Normalize B columns 
    if(which == 'Optical'):
        cols  = [col for col in df.columns if re.match(r'[0-9]+__B',col)]
        df.loc[:,cols] /= 1000 
        
    df = utils.target_remap(df , targetname , mapper =mapper, method=method_remap)
    
    
    df = smoothing(df, method= method_smooth,window=window,polyorder = polyorder)
    return df 
# COnsider using **kwargs
def pipeline_executable(first_arg,targetname = 'Crop_Type' ,which = 'Optical',mapper = {},impose_date=True,method_nulls = 'remcol' ,method_remap = 'Lazy',method_smooth= 'sav_gol',window=5,polyorder = 2):
    df = first_arg 
    df = preprocess(df,targetname = targetname , mapper = mapper , impose_date = impose_date , method_nulls = method_nulls , \
                        method_remap = method_remap ,method_smooth= method_smooth,window=window,polyorder = polyorder) 
        
    return df 
    
    
if __name__ == '__main__':
    fname = 'Data\\Interim\\CGDD\\FieldOptical_CGDD.csv'
    df = pd.read_csv(fname).drop(columns = ['Unnamed: 0'])
    
    impose_date = False
    method_nulls = 'interpol_pass' 
    df = pipeline_executable(df,impose_date = impose_date,method_nulls = method_nulls)

    df.to_csv('Data\\Interim\\Preprocessed\\FieldOptical_CGDD.csv',index=False)
    
        