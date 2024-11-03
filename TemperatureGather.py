# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:18:26 2024

@author: kashy
"""

'''
BUILD temperature data for the geometries (Harmonised.)

Input:
    
    1. AgERA5 temperature files. 
        This is the only script that accesses these files. 
    2. Harmonised geometries 
        Read if execution is in here, get as argument if its from pipeline
        (Pipline would probably call it multiple times for multiple years)
        
    
    
Output: 
     dataframes -- Harmonised_{version}_ max,min,22,23 etc. 
    If execution is here, save CSVs in Data/interim/post 
    If from pipeline, return the dataframes
    

'''

import xarray as xr
from glob import glob
import os
import re 
import pandas as pd
import numpy as np


from Generics import extract_centroid 

def get_vals_for_points(points,path,append):
    '''
    Path is a directory of all the 365 day values
    ''' 
    files = glob(os.path.join(path, '*.nc'))
    df = {}
    for f in files:
        filedata = xr.open_dataset(f)
        
        date =  re.search(r'AgERA5.*final' , f).group(0).split('_')[1]
        colname = date +'_'+append
        pointdatalist = []
        for lat,long in points:
            point_data = filedata.sel(lat=lat , lon = long ,method = 'nearest')
            pointdatalist.append(np.array(point_data.to_dataarray())[0][0])
        df[colname] = pointdatalist
    return df 
        
        


def gather(agera_directory,hm,year=2023):
    '''
    Directory should have subdirectory 'Temperature' 
    '''
    hm['points'] = hm['.geo'].apply(extract_centroid)
    ## the 4 paths are max, min of year,year-1 
    
    print("0%")
    
    path = agera_directory + '24 max '+str(year-1)
    dfmax22 = get_vals_for_points(hm['points'],path,'MAX')
    dfmax22 = pd.DataFrame(dfmax22)
    dfmax22['.geo'] = hm['.geo']
    dfmax22['Crop_Type'] = hm['Crop_Type']
    yield dfmax22 
    del dfmax22
    
    print("25%")
    
    path = agera_directory + '24 min '+str(year-1)
    dfmin22 = get_vals_for_points(hm['points'],path,'MIN')
    dfmin22 = pd.DataFrame(dfmin22)
    dfmin22['.geo'] = hm['.geo']
    dfmin22['Crop_Type'] = hm['Crop_Type']
    yield dfmin22 
    del dfmin22

    print("50%")
    path = agera_directory + '24 max '+str(year)
    dfmax23 = get_vals_for_points(hm['points'],path,'MAX')
    dfmax23 = pd.DataFrame(dfmax23)
    dfmax23['.geo'] = hm['.geo']
    dfmax23['Crop_Type'] = hm['Crop_Type']
    yield dfmax23 
    del dfmax23 

    print("75%")
    path = agera_directory + '24 min '+str(year)
    dfmin23 = get_vals_for_points(hm['points'],path,'MIN')
    dfmin23 = pd.DataFrame(dfmin23)
    dfmin23['.geo'] = hm['.geo']
    dfmin23['Crop_Type'] = hm['Crop_Type']
    yield dfmin23 
    del dfmin23
    



"#########################3"

if __name__ == '__main__':
    
    hm =  pd.read_csv('Data/input/harmonisedv5/harmonised_v5_2023_csv/hmv52023.csv')
    
    agera_directory = 'Data/input/Temperature/'
    dfmax22 , dfmin22, dfmax23 , dfmin23 = gather(agera_directory,hm,year=2023)
    
    dfmax22.to_csv('Data/interim/post/V5_dfmax22.csv')
    dfmin22.to_csv('Data/interim/post/V5_dfmin22.csv')
    dfmax23.to_csv('Data/interim/post/V5_dfmax23.csv')
    dfmin23.to_csv('Data/interim/post/V5_dfmin23.csv')
    
    