# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:19:02 2024

@author: kashy
"""

from utils import utils 

import pandas as pd 

def get_max_ndvi(df):
    df = utils.fix_column_syntax(df)
    timesteps , bands = utils.get_timesteps_bands(df , reg = r'[0-9]+__')
    cols = []
    d = {}
    timesteps_inv = {timesteps[i]:i for i in range(len(timesteps))}
    for time in timesteps:
        
        b4 , b8 = df[str(time)+'__B4'] , df[str(time)+'__B8']
        cols.append(str(time)+'__NDVI')
        d[str(time)+'__NDVI'] = (b8-b4) / (b8+b4)
    df = pd.concat([df , pd.DataFrame(d)],axis = 1)#
    df['max_NDVI'] = df[cols].max(axis=1)
    df['t_max_NDVI'] = df[cols].idxmax(axis=1).apply(lambda x: timesteps_inv[x.split('__')[0]])
    return df 
    
    
    
df = pd.read_csv('Data/Input/Satellite/Point_Optical.csv')
df = get_max_ndvi(df)
mn  = df['max_NDVI']
no = df['t_max_NDVI']