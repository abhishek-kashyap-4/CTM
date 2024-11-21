# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 05:42:08 2024

@author: kashy
"""


import pandas as pd
import EDA.EDA_Functions as ef
import matplotlib.pyplot as plt

def cloud_vs_temperature(df , hm_temperatures):
    merged = pd.merge(df,hm_temperatures , how = 'inner' , on = 'Unique_Id')
    merged = merged[[col for col in merged.columns if 'CLDPRB' in col or 'GDD' in col]] 
    dates  = [col.split('_')[0] for col in merged.columns if 'CLDPRB' in col]
    clds = [] 
    gdds = [] 
    for date in dates: 
        clds.append(merged[date+'_CLDPRB'].mean())
        gdds.append(merged[date+'_GDD'].mean()) 
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Darker grid style

    plt.scatter(clds,gdds)
    plt.xlabel('CLOUDS')
    plt.ylabel('GDD')
    plt.title('Correlation of Cloud Probability and GDD')
    
     
def tmax_vs_sowdate_vs_temperature(df , hm , hm_temperatures):
    '''
    This function should prove the effectiveness on gdd. 
    Both sowing date and average temperature of a field need to be correlated with variance of tmax ndvi.
    '''
    merged = pd.merge(df,hm_temperatures , how = 'inner' , on = 'Unique_Id')
    merged = merged[merged['Crop_Type'] == 'Maize']
    
    merged['avg_temp'] = merged[[col for col in merged if 'GDD' in col]].mean(axis=1)
    x = merged['t_max_NDVI'].values
    y = merged['avg_temp'].values
    plt.figure(300)
    plt.scatter(x , y)
    plt.show()
    
    df['Sow_Date'] = pd.to_datetime(df['Sow_Date'])
    earliest_date = df['Sow_Date'].min()
    
    merged['sowd'] =  (df['Sow_Date'] - earliest_date).dt.days

    merged = merged[merged['sowd'].notna()]
    
    plt.figure(32)
    plt.grid(color='gray', linestyle='--', linewidth=0.5) 
    x = merged['t_max_NDVI'].values
    y = merged['sowd'].values
    plt.scatter(x,y)
    plt.xlabel('T max ndvi')
    plt.ylabel('Sow_Date')
    plt.title('Correlation of sowdate and tmax vdvi')
    plt.show()


def pipeline_executable(first_arg,hm_temperatures = [],hm = []):
    df = first_arg 
    
    assert len(hm_temperatures) > 0 , "Please provide hm temperatures. "
    assert len(hm) >0 ,"Please provide hm"
    #cloud_vs_temperature(df,hm_temperatures)
    
    
    tmax_vs_sowdate_vs_temperature(df,hm ,hm_temperatures)
    
    
    
    
    

if __name__ == '__main__':
    df = pd.read_csv('Data/Interim/added/Optical_Field01.csv').drop(columns = ['Unnamed: 0'])
    hmt = pd.read_csv('Data/Interim/post/hm_v5_2023_temperatures.csv').drop(columns = ['Unnamed: 0'])
    hm = pd.read_csv("Data/Input/harmonisedv5/harmonised_v5_2023_csv/hmv52023.csv")
    
    pipeline_executable(df,hm_temperatures = hmt , hm = hm)