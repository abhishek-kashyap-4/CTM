# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:32:37 2024

@author: kashy
"""


import pandas as pd
from datetime import datetime ,timedelta
import numpy as np
from GlobalVars import crop_avg_sow_date

  

def GenerateCentroidTemperaturesCGDD(df,increment,method,startdate = '2023-01-01'):
    
    all_rows = []
    for rownum,row in df.iterrows():
        print(rownum*100/len(df))
        if(crop_avg_sow_date[row['Crop_Type']] ==-1):
            continue
        
        if(type(row['Sow_Date']) != str):
            start = datetime.strptime(crop_avg_sow_date[row['Crop_Type']],'%Y-%m-%d')
        else:        
            start = datetime.strptime(row['Sow_Date'],'%Y-%m-%d')
        if(method=='fixed'):
            start = datetime.strptime(startdate,'%Y-%m-%d')
        
        cummulation = 0
        nextstep = 0
        newrow = {'Start_date': start,'Crop_Type':row['Crop_Type'],\
                  'Unique_Id':row['Unique_Id'], '.geo':row['.geo'],
                  'Harv_Date':row['Harv_Date']}
        currdate = start
        while True:
            if(cummulation>=nextstep):
                newrow[nextstep] = cummulation
                newrow[str(nextstep)+'_DATE'] = currdate.strftime('%Y%m%d')
                nextstep+= increment
            currdate = currdate + timedelta(days=1)
            if(currdate.strftime('%Y%m%d')+'_GDD' not in centroid_temperatures.columns):
                break 
            cummulation += row[currdate.strftime('%Y%m%d')+'_GDD']
        all_rows.append(newrow)
    cgdd = pd.DataFrame(all_rows)
    return cgdd
    
if __name__ == '__main__':
    centroid_temperatures  = pd.read_csv("Data\\interim\\post\\temperature_V5_2023_centroids.csv",index_col=0)

    import yaml
    with open("config.yaml") as stream:
        try:
            dictionary = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    global_vars = dictionary['functions']['GenerateCentroidTemperaturesCGDD']
    
    method = global_vars['method']
    startdate = global_vars['startdate']
    increment = global_vars['increment']
    
    cgdd = GenerateCentroidTemperaturesCGDD(centroid_temperatures,increment,method,startdate = '2023-01-01')
    
    if(method=='fixed'):
        cgdd.to_csv('Data\\interim\\post\\cgdd_v5_fixed_'+startdate+'.csv',index=False) 
    else:
        cgdd.to_csv('Data\\interim\\post\\cgdd_v5_dynamic.csv',index=False) 
