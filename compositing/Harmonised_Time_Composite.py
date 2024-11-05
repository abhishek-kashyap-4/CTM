# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:12:53 2024

@author: kashy
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta



def aggregation(all_df , fromdate,increment,globalcounter,existing_dates,bands):
    fromdate_obj = datetime.strptime(fromdate,'%Y%m%d')
    dates = [fromdate]
    newdata = {}
    for i in range(increment): #if you had used -1 before, you should use +1 here 
        currdate_obj = fromdate_obj + timedelta(days=i)
        currdate = currdate_obj.strftime('%Y%m%d')
        if(currdate in existing_dates):
            dates.append(currdate)
    
    for band in bands:
        columns = [date+'__'+ band for date in dates]
        newdata[str(globalcounter)+'__'+band] =  all_df[columns].mean(axis=1)
    return newdata
 

def pipeline_executable(first_arg ,bands,fixed = False , fixed_date = '20230101',increment = 10):


    all_df = first_arg
    # A_K_ This is very dirty
    dated_columns = [val for val in all_df.columns if bands[0] in val]
    dates = sorted([val.split('__')[0] for val in dated_columns])
    if(fixed):
        dates = [val for val in dates if val>=fixed_date]
    newcompositecolumns = {}
    globalcounter = 0
    for i in range(0,len(dates),increment):
        fromdate = dates[i]
        newcompositecolumns |=  aggregation(all_df , fromdate,increment,globalcounter,dates,bands)
        globalcounter += 1
    df = pd.DataFrame(newcompositecolumns)
    
    # This method because for ex. sow date may not be there. 
    cols_to_copy = ['Unique_Id' , 'Sow_Date','Crop_Type']
    cols = [col for col in all_df.columns if col in cols_to_copy]  
    assert len(cols) >0 
    
    for col in cols:
        df[col] = all_df[col]
     #    df['Unique_Id'] = all_df['Unique_Id']
     #   df['Sow_Date'] = all_df['Sow_Date']
     #  df['Crop_Type'] = all_df['Crop_Type']
    return df 
    
if __name__ == '__main__':
    which = 'optical'

    if(which=='optical'):
        all_df  =  pd.read_csv('Data\\Input\\Satellite\\Optical0.1.csv').drop(columns = ['Unnamed: 0'])
        BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','CLDPRB']

    elif(which=='sar'):
        all_df  =  pd.read_csv('Data\\Input\\Satellite\\all_sar_restructured_cleaned.csv').drop(columns = ['Unnamed: 0'])
        BANDS = ["VV","VH","VV_VH"]
    
    fixed = False 
    increment = 10
    fixed_date = '20230101'
    df = pipeline_executable(all_df,BANDS,fixed = fixed , fixed_date = fixed_date,increment = increment)
    
    if(fixed):
        df.to_csv('Data\\Interim\\Satellite\\all_'+which+f'_{increment}day_fixedJan1.csv')
    else:
        df.to_csv('Data\\Interim\\Satellite\\all_'+which+f'_{increment}day.csv')
        
        
        

