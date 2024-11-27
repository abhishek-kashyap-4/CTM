# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:57:05 2024

@author: kashy
"""

import pandas as pd
import datetime
from datetime import timedelta

     
from GlobalVars import crop_base_temp 

from tk_printer import root,print_to_window
'''
Input
year-1 and year max and min files _ (4) 
and Harmonised to copy other columns 


If execution is here, save the files locally. If the execution is from the pipeline, 
Retrn the dataframe. 
'''

def check_and_merge(df1,df2):
    '''
    1. Assert that croptype and geo are the same. 
    2. drop intersecting columns 
        2.1 assert no interseccting columns? 
        2.2 maybe keep intersecting columns?
    3. merge dataframes.
    4. copy the geo and croptype of one of the datasets
    
    '''
    global global_variables
    assert sum(df1.Crop_Type != df2.Crop_Type) ==0, "Unequal croptypes"
    assert sum(df1['.geo'] != df2['.geo']) ==0, "Unequal geometries"
    
    croptype1 = df1.Crop_Type
    geo1 = df1['.geo']
    cols =  [col for col in df1.columns if col in df2.columns]
    df1   =  df1.drop(columns = cols) #Removes other columns
    df2 = df2.drop(columns = cols)
    
    df =pd.merge(df1,df2,left_index=True,right_index=True,how='inner')
    df['.geo'] = geo1
    df['Crop_Type'] = croptype1
    return df 

def generate(start,end,hm,columns_to_copy , append = ''):
    dfmax22 = pd.read_csv("Data\\interim\\post\\V5_dfmax22"+append+".csv").drop(columns=["Unnamed: 0"])
    dfmax23 = pd.read_csv("Data\\interim\\post\\V5_dfmax23"+append+".csv").drop(columns=["Unnamed: 0"])
    dfmin22 = pd.read_csv("Data\\interim\\post\\V5_dfmin22"+append+".csv").drop(columns=["Unnamed: 0"])
    dfmin23 = pd.read_csv("Data\\interim\\post\\\V5_dfmin23"+append+".csv").drop(columns=["Unnamed: 0"])
    
    dfmax22 = dfmax22.rename(columns = {'points':'.geo'})
    dfmax23 = dfmax23.rename(columns = {'points':'.geo'})
    dfmin22 = dfmin22.rename(columns = {'points':'.geo'})
    dfmin23 = dfmin23.rename(columns = {'points':'.geo'})
    
    df22 = check_and_merge(dfmax22,dfmin22)
    df23 = check_and_merge(dfmax23,dfmin23)
    
    df = check_and_merge(df23,df22)
    df['CROP_BASE_TEMP'] = [ crop_base_temp[ct] for ct in df.Crop_Type]
    
    curr = start 
    columns = {} 
    while(curr<end):
        curr += timedelta(days=1)
        currstr = curr.strftime('%Y%m%d')
        columnvals = []
        for rownum,row in df.iterrows():
            maxtemp = row[currstr+'_MAX'] - 273.15
            mintemp = row[currstr+'_MIN'] - 273.15
            basetemp = row['CROP_BASE_TEMP'] # in celcius
            if(mintemp<basetemp):
                columnvals.append(basetemp)
            else:
                columnvals.append( ( (maxtemp + mintemp)/2 ) -basetemp)
                
        columns[currstr+'_GDD']  = columnvals
    temperature_df = pd.DataFrame(columns)
    for col in columns_to_copy:
        temperature_df[col] = hm[col]
    
    
    return temperature_df    
        

global_variables = {}


def UNUSEDpipeline_executable(first_arg , *params):
    df = first_arg 
    dictionary['FUNCTIONS']['GenerateCentroidTemperatures']['columns_to_copy']
    
    # Maybe instead of this, we can have a function use 1 window no matter 
    #How many times it is called. 
    #I don't want to pass the window as an argument, since its supposed to be
    #excluded from the functionality
    # We will do this by - main creating a window for all functions that 
    #config says, and here you look at config dictionary and load that window.
    if(dictionary['FUNCTIONS']['GenerateCentroidTemperatures']['seperate_window']):
        
        import tkinter as tk
        window  = tk.Toplevel(root)
        window.title("GenerateCentroidTemperatures")
        print_to_window(window, locals())
    

if __name__=='__main__':
    
    start = datetime.date(2022,1,1)
    end = datetime.date(2023,12,30)
    
    
    annotated = False 
    if(annotated):
        harmonised2023 = pd.read_csv('Data\\input\\harmonisedv5\\Annotated_hm.csv')
        columns_to_copy = ['Unique_Id']
        temperaturedf = generate(start,end,harmonised2023,columns_to_copy,append ='_annotated')
        temperaturedf.to_csv("Data\\interim\\post\\hm_v5_2023_temperatures_annotated.csv")
    else:
        
        harmonised2023 = pd.read_csv('Data\\input\\harmonisedv5\\harmonised_v5_2023_csv\\hmv52023.csv')
        columns_to_copy = ['Unique_Id']
        temperaturedf = generate(start,end,harmonised2023,columns_to_copy)
        temperaturedf.to_csv("Data\\interim\\post\\hm_v5_2023_temperatures.csv")
    
    
    

    
    
    
    