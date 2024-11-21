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
        df = df.drop(columns = [col for col in df.columns if 'Unnamed' in col])
    except Exception as e:
        return False , None 
    return True , df
    

###################################################################################################### 
###################################################################################################### 
######################################################### Data related
def check_column_syntax(df, kind = 'generic' , reg = r'[0-9]+__',stricter = False):
    '''
    1. If kind is timestep:
        1.1 Columns with regex [0-9]{1,2}__ must be > 0 
        1.2 If stricter, they should be = timesteps * bands. 
        1.3 No other column starting with a digit should be present. 
    2. If Kind is date:
        2.1 Columns with refex [0-9]{8}__ must be >0
        2.2 If stricter, they should be = timesteps * bands 
        2.3 No other column starting with a digit should be present. 
    
    3. If kind is custom: 
         3.1 Columns with given regex must be >0 
         3.2 likewise 
         3.3 Likewise 
    4 If kind is genetic, regex is [0-9]+__
    '''
    if(kind == 'timestep'):
        regex = r'[0-9]{1,3}__' 
    elif(kind == 'date'):
        regex = r'[0-9]{8}__'
    elif(kind == 'custom'):
        regex = reg
    elif(kind == 'generic'):
         regex = r'[0-9]+__'
    else:
        raise Exception(f"Kind {kind} not recognized. Use among 'timestep' , 'date' ,'custom' ,'generic' ")
    
    cols = [col for col in df.columns if re.match(regex,col)]
    assert len(cols) >0 , "No columns matching the assumed regex."
    
    if(stricter):
        timesteps , bands = get_timesteps_bands(df,reg = regex)
        assert len(cols) == len(timesteps) * len(bands), "Strict column syntax check failed. (Tip- look at extra added features) "
        
    notcols = [col for col in df.columns if col not in cols]
    assert sum(1 for col in notcols if re.match(r'[0-9]',col)) ==0 , "Columns with Digit starting found that didn't match regex."

# A_K_ implement just_check.
def fix_column_syntax(df , from_re=r'[0-9]{8}_',impose_date = True,just_check = False):
    '''For now, just adding underscore to make it a doubleunderscore'''
    
    if(just_check):
        if(impose_date):
            check_column_syntax(df,kind = 'date')
        else:
            check_column_syntax(df,kind = 'custom',reg=from_re)
        return df 
    
    assert sum(1 for col in df.columns if '__' in col) ==0 , "Double underscore already found in columns. Change your code. (if you just want to check syntax, call check_column_syntax"

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
    assert len(d) >0  ,f"Your regex seems to be incorrect. For regex, <<{from_re}>> ;Affected columns is <<{len(d)}>>"
    return df.rename(columns = d)
            
            
def get_timesteps_bands(df , reg = -1 , check = False):
    
    #A_K_
    " ERROR - TIMESTEPS WERENT SORTED PROPERLY.CHANGED IT NOW."
    '''
    1. Get timesteps(sorted) and unique bands (only those with timesteps) that the dataframe has. 
    2. Check that all these bands have all timesteps. (Columns should either have all, or none of the timesteps)
     - timesteps HAVE to be sorted properly. (str sorts 11 before 2)
    '''
    if(reg==-1):
        warnings.warn("Regex wasn't set. Setting this to '[0-9]+__'. Note, this is not imposing YYYYMMDD and you should really send the regex you want. ",UserWarning)
        reg = '[0-9]+__'
        #The regex is numbers of any length followed by double underscore.
     
    
    # int() typecast works because we are using YYYYMMDD and not DDMMYYYY, and there's no leading zeroes.
    # It also works for redone timesteps. 
    timesteps , bands = zip(*[col.split('__') for col in df.columns if re.match(reg,col)])
    timesteps_unique , bands_unique = list(sorted(set( int(time) for time in timesteps))) , list(set(bands))
    timesteps_unique = [str(val) for val in timesteps_unique]
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

def get_timesteps_bands_columns(timesteps,bands):
    cols = []
    for band in bands:
        for time in timesteps:
            cols.append(str(time)+'__'+band)
    return cols
            
        

def get_timesteps(df,reg = -1,check = False):
    
    raise Exception("This function is discountinued. Use get_timesteps_bands instead. Remember to unpack both in return value.")
    

def check_df_nulls(df , method = 'row' , rowthres = 0.3 , colthres = 0.3 , rowtolerance = 0, coltolerance=0):
    """
    Check null values in df. 
    if method == row , check only for rows 
    if method == col, check only for cols 
    if method == both check both 
    """
    if(method == 'row' or method == 'both'):
        assert rowthres >= 0 and rowthres <= 1
        assert rowtolerance >=0 and rowtolerance <= df.shape[0]
        
        null_percentage = df.isnull().mean(axis=1)
        rows_exceeding_threshold = df[null_percentage > rowthres]
        if len(rows_exceeding_threshold) > rowtolerance:
            raise ValueError(f"Number of rows with null percentage > {rowthres*100}% exceeds tolerance of {rowtolerance}.")
        filtered_df_by_rows = df[null_percentage <= rowthres]
        
    if(method == 'col' or method == 'both'):
        assert colthres >= 0 and colthres <= 1
        assert coltolerance >=0 and coltolerance <= df.shape[1]
        
        null_percentage = df.isnull().mean(axis=0)
        columns_exceeding_threshold = df.loc[:, null_percentage > colthres]
        if len(columns_exceeding_threshold.columns) > coltolerance:
            raise ValueError(f"Number of columns with null percentage > {colthres*100}% exceeds tolerance of {coltolerance}.")
        filtered_df_by_cols = df.loc[:, null_percentage <= colthres]
        
    if(method == 'row'):
        return filtered_df_by_rows 
    elif(method == 'col'):
        return filtered_df_by_cols
    elif(method == 'both'):
        return -1
    else:
        raise Exception(f"Method {method} not recognized. Use 'row','col' or 'both' ")
    
def filter_df_nulls(df , method = 'remcol',subset = None, perc = 0.3):
    assert perc >=0 and perc<=1 ,"Perc should be b/w 0 and 1"
    if(method == 'remrow'):
        
        if(subset is not None):
            perc = 1 - perc # For df thres, it is the threshold of non nulls. 
            df_filtered = df.dropna(subset=subset,thresh = perc*df.shape[1])
            
        else:
            perc = 1- perc 
            df_filtered = df.dropna(thresh = perc*df.shape[1])
            perc = 1-perc
            df_filtered1 = check_df_nulls(df , method = 'row',rowthres = perc , rowtolerance = df.shape[0])
            assert df_filtered.equals(df_filtered1) , "check_df_nulls malfunction"
    elif(method == 'remcol'):
        df_filtered = check_df_nulls(df , method = 'col',colthres = perc , coltolerance = df.shape[1])
        colsremoved = [col for col in df if col not in df_filtered]
        print("Cols removed: \n",colsremoved)
    else:
        raise Exception(f"method {method} not recognized. Use remrow or remcol")
        
    new = df_filtered.shape[0] * df_filtered.shape[1] 
    orig = df.shape[0] * df.shape[1]
    print(f"Percentage of data removed: {((orig-new)/orig)*100} %")
    return df_filtered 
#default 30 percentage.
def OLD_filter_df_nulls(df , method = 'remcol',subset=None, perc = 30):
    if(method == 'remrow'):
        if subset is not None:
            df_filtered = df.dropna(subset=subset)
        else:
            df_filtered = df.dropna()
    elif(method == 'remcol'):
        threshold = len(df) * (perc / 100)
        # Drop columns that have more than the threshold of null values
        #df_filtered = df.dropna(axis=1, thresh=len(df) - threshold)
    
        cols_to_drop = df.columns[df.isnull().sum() > threshold]
        
        df_filtered = df.drop(columns=cols_to_drop)
        print("Columns removed: ")
        print(cols_to_drop)
    else:
        raise NotImplementedError()
    new = df_filtered.shape[0] * df_filtered.shape[1] 
    orig = df.shape[0] * df.shape[1]
    print(f"Percentage of data removed: {((orig-new)/orig)*100} %")
    return df_filtered 
    
def null_positions(df,rvc=False):
    if(rvc):
        print(f'Nulls -  {df.isnull().any().sum()} Columns , {df.isnull().any(axis=1).sum()} Rows.')
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
        df.loc[:,targetname] = [ mapper[val] for val in col]
        return df
    elif  method == 'Ignore':
        raise NotImplementedError
    elif method in ['Raise','Drop']:
        vals_in_mapper = set(mapper.keys())
        vals_in_col = set(df[targetname])
        assert vals_in_mapper == vals_in_col if method == 'Raise' else None 
        
    else:
        raise Exception(f'Method {method} not recognized')



def sampled_to_sampled(data,nsamples = 10,verbose=False):
  assert 'Unique_Id' in data.columns 
  Unique_Id = np.unique(data.Unique_Id)
  count=0
  lis = []
  for Unique_Id in Unique_Id:
    count+=1
    df = data[data.Unique_Id==Unique_Id]
    lis.append(df.iloc[:nsamples])
  print('Number of unique Fids:',count) if verbose else None

  return pd.concat(lis).reset_index(drop=True)

def sampled_to_reduced(data,by='mean',cutoff=10):
  '''
  Given N sampled data, basically convert it to reduced data.
  '''
  2/0, "didn't check"
  print('Please remember that reduction is only done for int64,float64 columns.')
  fids = np.unique(data.fid)
  count=0
  lis = []
  for fid in fids:
    d = {}
    count+=1
    df = data[data.fid==fid]
    #df = df.iloc[:10]
    for col in df.columns:
      if(df[col].dtype != np.int64 or df[col].dtype != np.float64):
        d[col] = df[col].iloc[0]
        continue
      if(by=='mean'):
        d[col] = df[col].mean()
      else:
        raise Exception('Only by mean is implemented so far')
    lis.append(d)
  print('Number of unique Fids:',count)
  return pd.DataFrame(lis)

    

def add_hm_features(df , hm , features = ['Crop_Type']):
    
    newrows  = []
    for rownum,row in df.iterrows(): 
        uid = row['Unique_Id'] 
        hmr = hm[hm.Unique_Id == uid].iloc[0]
        newrow = row 
        for feature in features:
            newrow[feature] = hmr[feature]
            
        newrows.append(newrow)
        
    return pd.DataFrame(newrows)


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





