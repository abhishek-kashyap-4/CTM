# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:14:19 2024

@author: kashy
"""
import re
import numpy as np
import pandas as pd


from utils.utils import fix_column_syntax , filter_df_nulls , null_positions , target_remap


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



def preprocess(df,targetname = 'CropType' ,mapper = {}, impose_date=True,method_nulls = 'remrow',subset=None ,method_remap = 'Lazy' ):
    df = fix_column_syntax(df,from_re = r'[0-9]{8}_' , impose_date = impose_date)
    
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
    fname = 'hm_optical_2023'
    df = pd.read_csv(fname+'.csv')
    impose_date = True
    df = pipeline_executable(df)
    1/0
    
        