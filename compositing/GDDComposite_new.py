# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:11:49 2024

@author: kashy
"""
from tqdm import tqdm 
import pandas as pd 
import numpy as np
from utils import utils 

import GlobalVars 
import itertools 
import datetime 

def CGDD(df,hm_temperatures , hm ,start_method = 'sowdate', fixed_date = '20230101',increment = 50):
    '''
  
    Inputs:
        1. Satellite Data  - hmV5 with Unique_Id
        2. Daily temperatures - hmv5 with Unique_Id
        3. hm that has croptype
    start_method = sowdate , fixed , notnull (untested) 
    
    I am going to iterate over Satellite Data, ensuring we get all the rows. 
        
    1. Take the data frame
    for row in iterrows:
        - define the dates where you have equal GDD intervals 
            - You can just get this from centroids.csv
        - Aggregate among these intervals 
        
    Decision - do I compute centroidscsv separately, or include it in iterrows?
        - Only daily temperatures are pre computed. 
        
    
    
    '''
    #timesteps (dates) , will be sorted.
    
    
    hm_temperatures = utils.fix_column_syntax(hm_temperatures , from_re = r'[0-9]+_',impose_date = True)
    gdd_timesteps , gdd_bands = utils.get_timesteps_bands(hm_temperatures)
    assert len(gdd_bands) == 1
    satellite_timesteps , satellite_bands = utils.get_timesteps_bands(df)
    enddate = satellite_timesteps[-1]
    enddate = datetime.datetime.strptime( enddate, '%Y%m%d')
    
    startdate_from_satellite = satellite_timesteps[0]
    
    newrows = []
    anchors = []
    
    for rownum ,row in tqdm(df.iterrows() , total = df.shape[0]):
        
        uid = row['Unique_Id']  
        hm_temperature_row = hm_temperatures[hm_temperatures.Unique_Id == uid]
        hm_row = hm[hm.Unique_Id == uid] 
        
        assert len(hm_temperature_row) == 1 , f' UID {uid} {len(hm_temperature_row)}'
        assert len(hm_row) == 1  , f' UID {uid} {len(hm_row) }'
        hm_temperature_row = hm_temperature_row.iloc[0]
        hm_row = hm_row.iloc[0]
        
        
        # Getting the Start Date.
        #A_K_ IMP - right now Sow_Date is like yyyy/mm/dd. explicitly changing it.
        ################################################################################33333
        if(start_method == 'sowdate'):
            startdate = hm_row['Sow_Date']
            if(type(startdate) != str):
                startdate = GlobalVars.crop_avg_sow_date[hm_row['Crop_Type']]
                if(startdate ==-1):
                    startdate = startdate_from_satellite 
                else:                      
                    startdate = datetime.datetime.strptime(startdate,'%Y-%m-%d').strftime('%Y%m%d')
            else:
                startdate = datetime.datetime.strptime(startdate,'%Y-%m-%d').strftime('%Y%m%d')
                
        elif(start_method == 'fixed'):
            startdate =  fixed_date
        elif(start_method == 'notnull'):
            startdate = hm_row['Sow_Date']
            startdate = datetime.datetime.strptime(startdate,'%Y-%m-%d').strftime('%Y%m%d')
            if(type(startdate) != str):
                continue 
        else:
            raise Exception("Start Method not recognized.")
            
        startdate = datetime.datetime.strptime( startdate, '%Y%m%d')
        
        #Getting Cumulatted GDD
        ################################################################################33333
        gdd_timesteps_row = []
        for date in gdd_timesteps:
            date_as_date = datetime.datetime.strptime( date, '%Y%m%d')
            if(date_as_date <startdate or date_as_date >enddate):
                continue 
            gdd_timesteps_row.append(date) 
            
        daily_temperatures = [hm_temperature_row[date+'__GDD'] for date in gdd_timesteps_row]
        
        ### not required 
        #cumulated_temperatures = list(itertools.accumulate(daily_temperatures))
        
        # Get anchor points
        ################################################################################33333
        anchor_points = [startdate.strftime('%Y%m%d')]
        cumulation = 0 
        
        for date,gdd in zip(gdd_timesteps_row , daily_temperatures):
            cumulation += gdd 
            if(cumulation > increment):
                anchor_points.append(date) 
                cumulation = 0 
        anchor_points.append(enddate.strftime('%Y%m%d'))
        
        # Get aggregated sublists for satellite_timesteps 
        # You can't use keys (anchors w/o start) as column names as anchors are different for each row. 
        # You'd need to save index, and save anchors separately.
        ################################################################################33333
        anchors.append([uid]+anchor_points[1:])
        aggregated_sublists = {}
        for date in satellite_timesteps: 
            date_as_date = datetime.datetime.strptime(date,'%Y%m%d')
            if(date_as_date <startdate or date_as_date>enddate ):
                continue 
            
            for i in range(len(anchor_points)) :
                if(date_as_date < datetime.datetime.strptime(anchor_points[i],'%Y%m%d')):
                    break 
            if(i not in aggregated_sublists):
                aggregated_sublists[i] = [date]
            else:
                aggregated_sublists[i].append(date)
        
        # Aggregate all bands based on aggregated sublists
        # Use key-1 since i=0 isn't being used. 
        ################################################################################33333
        aggregated_df = {}
        aggregated_df['Unique_Id'] = uid 
        
        for key in aggregated_sublists:
            dates = aggregated_sublists[key]
            for band in satellite_bands:
                aggregated_df[str(key-1)+'__'+band] = row[[date+'__'+band for date in dates]].mean() 
        
        newrows.append(aggregated_df)
    
    return pd.DataFrame(newrows) , pd.DataFrame(anchors)
        
            

def pipeline_executable(first_arg , hm_temperatures = [] , hm = [] , increment = 75,anchor_save = -1,start_method = 'sowdate', fixed_date = '20230101'):
    df = first_arg 
    
    assert len(hm_temperatures) > 0 , "Provide hm_temperatures" 
    assert len(hm) > 0 , "Provide hm " 
    final , anchors = CGDD(df , hm_temperatures , hm , increment = increment,start_method = start_method, fixed_date = fixed_date) 
    anchors = anchors.rename(columns = {'0':'Unique_Id'})
    anchors.to_csv(anchor_save)
    return final 
    
    
    
    

if __name__ == '__main__':
    
    
    hm_temperatures  = pd.read_csv("Data\\Interim\\post\\hm_v5_2023_temperatures.csv",index_col=0)
    hm = pd.read_csv('Data\Input\harmonisedv5\harmonised_v5_2023_csv\hmv52023.csv')
    df = pd.read_csv('Data\Input\Satellite\Field_SAR.csv')
    
    df = utils.fix_column_syntax(df,from_re = r'[0-9]+_')
    final , anchors = pipeline_executable(df,hm_temperatures , hm , increment = 50, anchor_save = 'Data\Interim\CGDD\FieldSAR_anchors.csv')
    
    final.to_csv('Data\Interim\CGDD\FieldSAR_CGDD.csv')
    
    
    