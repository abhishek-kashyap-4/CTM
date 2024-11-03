# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:14:19 2024

@author: kashy
"""
import re
import numpy as np
import pandas as pd


from utils.utils import fix_column_syntax , filter_df_nulls , null_positions , target_remap , get_timesteps_bands


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

def null_interpol(df):
    timesteps , bands = get_timesteps_bands(df)
    count = 1
    for band in bands:
        print(count/len(bands))
        count += 1
        cols = [str(time)+'__'+ band for time in timesteps]
        df = kalman_fill(df,cols)
    return df 



def preprocess(df,targetname = 'CropType' ,mapper = {}, impose_date=True,method_nulls = 'remrow',subset=None ,method_remap = 'Lazy' ):
    df = fix_column_syntax(df,from_re = r'[0-9]+_' , impose_date = impose_date)
    
    df = filter_df_nulls(df , method= method_nulls,subset=subset)
    nullp = null_positions(df)
    
    assert len(nullp) == 0 ,"Null values detected."
    df = target_remap(df , targetname , mapper =mapper, method=method_remap)
    return df 
    


def pipeline_executable(first_arg,targetname = 'CropType' ,mapper = {},impose_date=True,method_nulls = 'remrow',subset=None ,method_remap = 'Lazy'):
    df = first_arg 
    df = preprocess(df,targetname = targetname , mapper = mapper , impose_date = targetname , method_nulls = method_nulls , \
                        subset = subset , method_remap = method_remap) 
    return df 
    
    
if __name__ == '__main__':
    fname = 'Data\\Interim\\CGDD\\cgdd_Beta002.csv'
    df = pd.read_csv(fname)
    impose_date = True
    
    print(len(null_positions(df)))
    df = null_interpol(df)
    print(len(null_positions(df)))
    
    
    2/0
    df = pipeline_executable(df)
    1/0
    
        