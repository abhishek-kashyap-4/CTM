# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:14:19 2024

@author: kashy
"""
import re
import numpy as np
import pandas as pd


# Don't use Cloudmask as an input for cloudfillers since the nan values could have come from elsewhere.

def cloudfill_on_3d(array,feature_index,threshold):
    '''
    Use this to perform cloudfilling on 3d numpy version. 
    
    '''
def cloudfill_by_clustering(df,feature_pattern,threshold=75):
    '''
    Clustering algorithm for cloudfill:
        Assuming that pixels can be clustered with the available bands,
        1. Use Clustering method to cluster all pixels. Optionally use the reduced version of the data.
        2. Get the neighbours of the current 
    '''


def cloudfill(df ,feature_pattern,obj,obj_defined = True,threshold=75):
    '''
    for every timestep, use the feature pattern to get all cloud features
    expecting the columns to be of YYYYMMDD__BAND_Name. 
    But don't assert here as there are other columns (ground truth for instance)
    '''
    clouds = [col for col in df.columns if re.search(feature_pattern,col)]
    timesteps = [col.split('__')[0] for col in clouds]
    
    assert timesteps == obj.timesteps , "Timestep mismatch in def cloudfill."
    
    '''
    1. Loop through every row
    2. Remove nan timesteps 
    3. Wherever cloud band is more than threshold, 
        3.1 find nearest n times with less than threshold 
        3.2 find mean
        3.2 replace 
        
    '''
    newdf = []
    for rownum,row in df.iterrows():
        newrow = []
        newdf.append(newrow)
        
    return pd.DataFrame(newdf)

def cloudmask(df, feature_pattern, threshold = 75,dropfeature=True):
    '''
    Feature_Pattern has the cloud feature. Get the prefix timestep from it. 
    Where the value is more than threshold, mask the values. 
    
    '''
    clouds = [col for col in df.columns if re.search(feature_pattern,col)]
    timesteps = list(set([col.split('__')[0] for col in clouds]))
    
    #A_K_ Dirty hack
    cols = [col for col in df.columns if timesteps[0] in col and feature_pattern not in col]

    for timestep in timesteps:
        reconstructed = timestep +'__'+feature_pattern
        assert  reconstructed in clouds , f'{reconstructed} not found in Clouds'
        clouds  = [col for col in clouds if col!=reconstructed]
        df.loc[df[reconstructed]>threshold , cols]  = np.nan
        
    
    if(len(clouds)>0):
        raise Exception("Something doesn't make sense here. Perhaps some duplicate columns") 

    return df 

def pipeline_executable(first_arg, reg =r'[0-9]{8}_',feature_pattern= 'CLDPRB',method = 'Mask'):
    
    if(method != 'Mask'):
        raise NotImplementedError
    df = first_arg
    from utils.utils import fix_column_syntax
    df = fix_column_syntax(df,from_re = reg)
    
    newdf = cloudmask(df ,feature_pattern,threshold=75)
    #newdf.to_csv('Data/Interim/Cloud/Optical_Cloudfilled.csv')
    return newdf
    
        
if __name__ == '__main__':
    
    fname = 'Data/Input/Satellite/Optical.csv'
    df = pd.read_csv(fname)
    newdf = pipeline_executable(df)
    newdf.to_csv('Data/Interim/Cloud/Optical_Cloudfilled.csv')
        