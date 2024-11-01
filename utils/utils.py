# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:47:37 2024

@author: kashy
"""
import re
import warnings
import inspect
from datetime import datetime 


import pandas as pd
import numpy as np

import GlobalVars


###################################################################################################### 
####################################### GENERICS

class AK_Logger:
    def __init__(self):
        # Automatically get the name of the calling function
        frame = inspect.currentframe().f_back
        self.function_name = frame.f_code.co_name
        self.verbose = []

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.verbose.append(f"{timestamp} - {message}")

    def save_and_clear(self):
        # Save to a file named after the function that created this logger
        file_path = f"{self.function_name}.log"
        with open(file_path, 'w') as file:
            file.write("\n".join(self.verbose))
        self.verbose.clear()

# Usage - 1) log = AK_logger() 2) log.log('What is this') 3) log.save_and_clear()


####################################################################################################3
####################################### I/O
def safe_load(fname,ftype='csv'):
    '''
        Return df if fname is found along with message.
        If not, return message and None. 
    '''
    if(ftype != 'csv'):
        raise NotImplementedError
    try:
        df = pd.read_csv(fname)
    except Exception as e:
        return False , None 
    return True , df
    

###################################################################################################### 
###################################################################################################### 
######################################################### Data related

def fix_column_syntax(df , from_re=r'[0-9]{8}_',impose_date = True):
    '''For now, just adding underscore to make it a doubleunderscore'''
    
    if sum(1 for col in df.columns if '__' in col)>0 :
        print('Double underscore already found in columns. Change your code. ')
        return df 
    
    if(impose_date):
        cols = [col for col in df.columns if re.match(r'[0-9]', col) ]
        checked = [col for col in cols if re.match(r'[0-9]{8}',col)]
        assert len(cols) == len(checked) , "some column with number starting don't seem to have a date. If you're calling with timesteps, make sure to use impose_date = False"
        
    # A_K_ be even more strict by taking global year and imposing between year date. 
    

    d =  {}
    for col in df.columns:
        if(re.match(from_re,col)):
            newname = re.sub(from_re, r'\g<0>_', col)
            d[col] = newname 
    assert len(d) >0 ,"Your regex seems to be incorrect."
    return df.rename(columns = d)
            
            
def get_timesteps_bands(df , reg = -1 , check = False):
    '''
    1. Get timesteps and unique bands (only those with timesteps) that the dataframe has. 
    2. Check that all these bands have all timesteps. (Columns should either have all, or none of the timesteps)
    '''
    if(reg==-1):
        warnings.warn("Regex wasn't set. Setting this to '[0-9]+__'. Note, this is not imposing YYYYMMDD and you should really send the regex you want. ",UserWarning)
        reg = '[0-9]+__'
        #The regex is numbers of any length followed by double underscore.
    else:
        raise NotImplementedError('This functionality isn not implement yet. Just use default (reg=-1) for now.')
     
    
    # int() typecast works because we are using YYYYMMDD and not DDMMYYYY, and there's no leading zeroes.
    # It also works for redone timesteps. 
    timesteps , bands = zip(*[col.split('__') for col in df.columns if re.match(reg,col)])
    timesteps_unique , bands_unique = list(sorted(set(timesteps))) , list(set(bands))
    
    from collections import Counter 
    #A_K_  check this
    if(check):
        tu ,bu = len(timesteps_unique) ,len(bands_unique)
        timesteps_counter , bands_counter = Counter(timesteps) , Counter(bands)
        
        for timestep in timesteps_unique: 
            assert timesteps_counter[timestep] == bu ,"Some timesteps seem to be missing some bands."
        for band in bands_unique:
            assert bands_counter[band] == tu , "Some bands seem to be missing all timesteps."
    
    return timesteps_unique, bands_unique 

def get_timesteps(df,reg = -1,check = False):
    '''
    Get list of times (dates) that a dataframe has. 
    Optionally check that all columns have values for either all or none of the timesteps. 
    
    '''
    
    
    
    raise Exception("This function is discountinued. Use get_timesteps_bands instead. Remember to unpack both in return value.")
    
    print("Make sure the regex is called differently. Regex for imposing and not imposing YYYYMMDD is different, and should be.")
    if(reg==-1):
        warnings.warn("regex wasn't set. Setting this to '[0-9]+__. Note, this is not imposing YYYYMMDD and you should really send the regex you want. ",UserWarning)
    else:
        raise NotImplementedError('This functionality isn not implement yet. Just use default (reg=-1) for now.')
        
    #The regex is numbers of any length followed by double underscore.
    # int() typecast works because we are using YYYY, and there's no leading zeroes
    
    cols = sorted(set([int(col.split('__')[0]) for col in df.columns if(re.match(r'[0-9]+__',col))]))
    
    #A_K_ implement this. 
    if(check):
        raise NotImplementedError
    
    return cols 

def sample_df():
    pass 
def filter_df_nulls(df , method = 'remcol',subset=None):
    if(method == 'remrow'):
        if subset is not None:
            df_filtered = df.dropna(subset=subset)
        else:
            df_filtered = df.dropna()
        return df_filtered
    elif(method == 'remcol'):
        removed_cols = df.columns[df.isnull().any()].tolist()
        df_cleaned = df.drop(columns=removed_cols)
        print("Columns removed: ")
        print(removed_cols)
        return df_cleaned
    else:
        raise NotImplementedError()
    
def null_positions(df):
    ''' Get the null positions (index,colname) of a dataframe'''
    null_positions = list(df.isnull().stack()[df.isnull().stack()].index)
    return null_positions

def target_remap(df , targetname , mapper={}, method='Lazy'):
    '''
    Remap the target column with the mapper. 
    If method == Lazy , expect the mapper to be perfect 1 on 1. 
    If method = Drop, columns absent in the mapper should be dropped.
    If method = Ignore, leave the columns as is. 
    If method = Raise , columns are not supposed to be absent from the mapper.
    '''
    if len(mapper) == 0 :
        mapper = GlobalVars.target_remap
        
    if method == 'Lazy':
        col= df[targetname] 
        df[targetname] = [ mapper[val] for val in col]
        return df
    elif  method == 'Ignore':
        raise NotImplementedError
    elif method in ['Raise','Drop']:
        vals_in_mapper = set(mapper.keys())
        vals_in_col = set(df[targetname])
        assert vals_in_mapper == vals_in_col if method == 'Raise' else None 
        
    else:
        raise Exception(f'Method {method} not recognized')


    


###################################################################################################### 
###################################################################################################### 
######################################### Remote Sensing 
import json
from shapely.geometry import shape

def extract_centroid(geo_json_str):
    '''Centroid from a json geometry like that of GEE    '''
    geo_json = json.loads(geo_json_str)
    geom = shape(geo_json)
    centroid = geom.centroid
    return (centroid.y, centroid.x)


###################################################################################################### 
###################################################################################################### 
######################################################### ML 





