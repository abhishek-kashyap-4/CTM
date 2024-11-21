# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:40:12 2024

@author: kashy
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:32:37 2024

@author: kashy
"""


import pandas as pd
from datetime import datetime ,timedelta
import numpy as np
from GlobalVars import crop_avg_sow_date



get_base_temp = {
}

def get_daily_temperatures(hm_daily , uid , asindex=False):
    '''
    Given a UID, get daily temperatures from the file. 
    you wont know start and end dates here, just return everything. 
    
    '''
    if(asindex):
        
        row =  hm_daily.loc[uid] 
        print(type(row))
    else:
        row = hm_daily[hm_daily.Unique_Id == uid]
        assert len(row) == 1 
        row = row.iloc[0]
    
    print(row)
    1/0



def CGDD(df,hm_temperatures , hm , increment = 50):
    '''
  
    Inputs:
        1. Satellite Data  - hmV5 with Unique_Id
        2. Daily temperatures - hmv5 with Unique_Id
        3. hm that has croptype
    
    I am going to iterate over Satellite Data, ensuring we get all the rows. 
        
    1. Take the data frame
    for row in iterrows:
        - define the dates where you have equal GDD intervals 
            - You can just get this from centroids.csv
        - Aggregate among these intervals 
        
    Decision - do I compute centroidscsv separately, or include it in iterrows?
        - Only daily temperatures are pre computed. 
        
    
    
    '''

    newrows = {} 
    
    for rownum ,row in tqdm(df.iterrows() , total = df.shape[0]):
        
        uid = row['Unique_Id']
        hm_temperature_row = hm_temperatures[hm_temperatures.Unique_Id == uid]
        hm_row = hm[hm.Unique_Id == uid] 
        
        print(type(hm_temperature_row))
        print(type(hm_row)) 
        1/0
        
        
        croptype = row['Crop_Type']
        geometry = row['geometry']
        
        base_temp = get_base_temp[croptype]
        
        
        daily_temps = get_daily_temps() #Don't know if I need to sent centroid or polygon 
        # dictionary of date and value 
        # Alternately, get 2 dicts for day and night 
        
        cummulation = 0 
        threshold = 0
        single_row = {}
        for date in daily_temps:
            # Add cummulation 0 in the beginning 
            if(theshold >= increment):
                # we fill make a decision about adding the previous non nan value, or taking a window and aggregating.
                single_row[date] 
            
            
        #for i in range(0,maxcgdd , increment):
        

  

def GenerateCentroidTemperaturesCGDD(centroid_temperatures,increment,method,startdate = '2023-01-01'):
    
    all_rows = []
    for rownum,row in centroid_temperatures.iterrows():
        print(rownum*100/len(centroid_temperatures))
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
    
    
def pipeline_executable(first_arg,method = 'fixed' , increment =25,startdate = '2023-01-01'):
    centroid_temperatures = first_arg 
    cgdd = GenerateCentroidTemperaturesCGDD(centroid_temperatures,increment=increment,method=method, startdate= startdate)
    return cgdd
    
if __name__ == '__main__':
    centroid_temperatures  = pd.read_csv("Data\\interim\\post\\temperature_V5_2023_centroids.csv",index_col=0)

    method = 'fixed'
    startdate = '2023-01-01'
    increment = 25
    
    cgdd = GenerateCentroidTemperaturesCGDD(centroid_temperatures,increment,method,startdate = startdate)
    
    if(method=='fixed'):
        cgdd.to_csv('Data\\interim\\post\\cgdd_v5_fixed_'+startdate+'.csv',index=False) 
    else:
        cgdd.to_csv('Data\\interim\\post\\cgdd_v5_dynamic.csv',index=False) 
      
'###############################################################################33333333'

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:40:12 2024

@author: kashy
"""




# %%

###########################################################################################


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:31:10 2024

@author: kashy
"""

'''
This should cover everything from Loading all_df_restructured to Getting GDD scale and time scale
with variable, custom increments. 

I will make a decision of also including the code to restructure here, but not for now. 

Also give a few rows from df_restructured, be able to get corresponding rows for 
1. gdd scale and 
2. Time scale. 
I plan to use this for both single rows and subsets like Sow_Date != NA


The input would be all_df_restructured, which = optical or sar (since bands is dependant, and also sys index.)
and cgdd dates. 
also load hm2023 as it has all the fields including errors and abandoned ones. 
Other dependacies are BANDS, Exception crops etc.

'''

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from datetime import datetime,timedelta
import math
import seaborn as sns 
import re
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import utils
# %%
####################################FUCNTIONS###################################################
"##### Level 1 functions ##"
def aggregate_bands(df,columns):
    '''
    This should take the dataframe of multiple rows, 
    aggregate the BANDS, and copy the rest. 
    We can use this functionality to aggregate points form the same field. 
    '''
    newdf = df.iloc[0].copy()
    newdf[columns] = df[columns].mean()
    return pd.DataFrame(newdf).T

        
def getdates(row,maxcgdd,increment, increment2=0,method='static',):
    '''
    Methods:
        1. static: Use static increment from zero 
        !! (update this in cgdd pls.)
        2. Season: Use 1 for winter, 2 for summer.
        3. gradient: yet to do.This needs global average temperature. 
    '''
    cols = []
    gdds = []
    monthly_avg_temps = {'01' : -3, '02' : -2, '03' : 2,
                         '04' : 10,'05' : 16,'06' : 20,
                         '07' : 22,'08' : 21,'09' : 16,
                         '10' : 10, '11' : 4,'12' : -1,}
    
    if(method=='static'):
        gddrange = range(0+increment,maxcgdd+1,increment)
    elif(method=='season'):
        raise NotImplementedError
    elif(method=='grdient'):
        raise NotImplementedError
    else:
        raise ValueError
        

    #1st zero is absent, so doing 0+ increment
    for gdd in gddrange: 
        
        if( str(gdd) in row and not math.isnan(row[str(gdd)])):
            cols.append(str(gdd)+'_DATE')
            gdds.append(str(gdd))  
    return row[cols] , row[gdds]
# %%

"##### Level 2 functions ##"
def intepol_singleline(row ,index,date,bands , method='lookbehind',n=3,extreme=20):
    
    '''
    ASSUMING THE DATES ARE SORTED.
    
    This is slightly a misnomer as this function isn't for a full line, but for 1 date.
    method: 
            1. lookbehind  - Last available date. Go until extreme.
            2. lookahead - First date since. Go until extreme.
            3. closest - Closest date. 
            4. reduce_mean_n - Use mean for the nearest n dates. 
            5. reduce_median_n- Median for the nearest n dates. 
            6. reduce_mean_extreme - Mean for +extreme to -extreme
            7. reduce_median_extreme - Median for + extreme to -extreme
            8. reduce_mean_extreme_left - (6), only for -extreme to present
            9. reduce_mean_extreme_right - (6), only for present to +extreme 
    Looking behind makes sense 
    because at the date, the CGDD is ATLEAST the amount. 
    So there's some excess.
    
    '''
    if(method in ('reduce_mean_n','reduce_median_n')):
        raise NotImplementedError
    
    def get_left_right(datetouse,extreme):
        lefts = []
        rights = []
        for i in range(1,extreme):
            left = datetouse - timedelta(days=i)
            left = left.strftime('%Y%m%d')
            if(left+'__'+bands[0] in row):
                s = sum([math.isnan(row[left+'__'+band]) for band in bands]) 
                if(s==0):
                    lefts.append(left)
                else:
                    assert  s==len(bands) or s==len(bands)-1, f"Null value inconsistency for date {left} with nulls being {s}."
            right = datetouse + timedelta(days=i)
            right = right.strftime('%Y%m%d')
            if(right+'__'+bands[0] in row):
                s = sum([math.isnan(row[right+'__'+band]) for band in bands]) 
                if(s==0):
                    rights.append(right)
                else:
                    assert  s==len(bands) or s== len(bands)-1, f"Null value inconsistency for date {right} with nulls being {s}"
        return lefts,rights
    
    def date_validity(row,datetouse):
        if(datetouse+'_'+bands[0] in row ):
            s = sum([math.isnan(row[datetouse+'__'+band]) for band in bands]) 
            if(s==0):
                return True
            assert s == len(bands)
        return False
        
    #####
    datetouse_date = datetime.strptime(str(int(date)) , '%Y%m%d')
    datetouse = datetouse_date.strftime('%Y%m%d')
    
    if(method in ('lookbehind','lookahead','closest')):
        lefts,rights = get_left_right(datetouse_date,extreme =extreme) #You should give 2 for one value.
        if(datetouse+'__'+bands[0] in row ):
            if(date_validity(row,datetouse)):
                return {str(index)+'__'+band : row[datetouse+'__'+band] for band in bands}
        if(method =='lookbehind'):
            assert len(lefts) !=0 , f"For extreme {extreme}, lefts is zero for date, {date}"
            return {str(index)+'__'+band : row[lefts[-1]+'__'+band] for band in bands}
        elif(method=='lookahead'):
            return {str(index)+'__'+band : row[rights[0]+'__'+band] for band in bands}
        elif(method=='closest'):
            #Heres the thing, you can just send closest date,
            #and that will be easy, but technically speaking
            #we want the closest date to the correct CGDD, 
            #Which isn't the same as closest to datetouse.
            #You need a cgdd of not just 25 increments, but everything. 
            if(len(lefts)==0 and len(rights)==0):
                tqdm.write('wattass')
                return {} #What can men do against such reckless hate?
            if(len(lefts)==0):
                return {str(index)+'__'+band : row[rights[0]+'__'+band] for band in bands}
            if(len(rights)==0):
                return {str(index)+'__'+band : row[lefts[-1]+'__'+band] for band in bands}
            
            if(datetouse_date - datetime.strptime(lefts[-1], '%Y%m%d') < \
               datetime.strptime(rights[0], '%Y%m%d')-datetouse_date):
                
                return {str(index)+'__'+band : row[lefts[-1]+'__'+band] for band in bands}
            else:
                return {str(index)+'__'+band : row[rights[0]+'__'+band] for band in bands}

    elif('reduce' in method):
        lefts,rights = get_left_right(datetouse_date,extreme)
        dates = []
        #Weird code below I know, but its right. 
        #(Given you didnt add any more methods that interfere; its 9 currently)
        if('left' not in method):
            dates += rights
        if('right' not in method):
            dates += lefts
        if(date_validity(row,datetouse)):
            dates += [datetouse]
        if(len(dates)==0):
            return {} 
        retval = {}
        if('mean' in method):
            for band in bands:
                retval[str(index)+'__'+band] = np.array([row[date+'__'+band] for date in dates]).mean()
        elif('median' in method):
            for band in bands:
                retval[str(index)+'__'+band] = np.median(np.array([row[date+'__'+band] for date in dates]))
        else:
            36/0 , 
        return retval
            
                
def gdd_interpol(all_df, hm , cgdd,bands ,interpol_method = 'closest',increment = 50,extreme = 5,  maxcgdd = 5000, startdate = '2023-01-01',date_method = 'fixed',verbose =False):
    '''
    Given the 3 dataframes, return df with cgdd axis.
    Neither of all_df , cgdd need to have all the points. So loop with hm.
    '''
    gdd_interpol_verbose = {
        'absolutetotalskips' : 0,
        'sanity' : 0,
        'progress' : 1
        }
    
    # A_K_ The check here is important for the script.
    timesteps , COLUMNS = utils.get_timesteps_bands(all_df , reg = '[0-9]{8}__', check = True)
    
    assert set(COLUMNS) ==set(bands)
    
    unique_ids = np.unique(hm.Unique_Id)
    #unique_ids = [fid for fid in unique_ids if fid in cgdd['Unique_Id'].values]
                  
    unique_ids = [fid for fid in unique_ids if fid in cgdd['Unique_Id'].values and fid in all_df['Unique_Id'].values]

    newrows = []
    for fid in tqdm(unique_ids):
    #for fid in unique_ids:
        
        #print(gdd_interpol_verbose['progress']/len(unique_ids)) if verbose else -1
        gdd_interpol_verbose['progress'] += 1
        
        hmrow = hm[hm.Unique_Id==fid].iloc[0]
        datesdf = cgdd[cgdd['Unique_Id']==fid]
        df = all_df[all_df['Unique_Id']==fid]
        
        #sardf = aggregate_bands(sardf,COLUMNS)
        assert len(datesdf)==1  , f'FID {fid},datesdf size is {len(datesdf)}'
        #assert len(df) in (1,2) , (fid , len(df))
        
        dates ,gdds= getdates(datesdf.iloc[0],maxcgdd=maxcgdd,increment=increment)

        
        for rownum,row in df.iterrows():
            newrow = {}
            gdd_interpol_verbose['sanity'] += 1
            for index,date in enumerate(dates):
                newrow |= intepol_singleline(row ,index,date,bands,method= interpol_method,extreme=extreme)
            
            newrow['Crop_Type'] = hmrow['Crop_Type']
            newrow['Unique_Id'] = hmrow['Unique_Id']
            if(date_method=='fixed'):
                newrow['Sow_Date'] = startdate
            else:
                newrow['Sow_Date'] = hmrow['Sow_Date']
            newrows.append(newrow)
    print("Sanity = ", gdd_interpol_verbose['sanity'])
    
    
    return pd.DataFrame(newrows)
            


# %%
#########################################################################

def pipeline_executable(first_arg,hm = [] , cgdd = [],bands = [],interpol_method = 'closest',increment = 50,extreme = 5, maxcgdd = 5000,date_method = 'fixed',  verbose =True  ):
    
    df = first_arg 
    
    if(len(hm)<1):
        raise Exception("Please provide Hm dataframe.")
    if(len(cgdd)<1):
        raise Exception("Please provide CGDD dataframe.(Execute GenerateCentroidTemperaturesCGDD) ")
        
    if(len(bands)<1):
        raise Exception("Provide bands. ")
    
    if('Unique_Id' not in hm or 'Unique_Id' not in df or 'Unique_Id' not in cgdd):
        raise Exception("GDD scrip expects Unique_Id to be present in all dataframes.")
           

    #A_K_ precarious. be careful. 
    maxcgdd = max(int(col.split('_')[0]) for col in cgdd.columns if 'DATE' in str(col) )
    
    ## Quite important to do this.
    cgdd.columns = cgdd.columns.astype(str)
    
    utils.check_column_syntax(df , kind = 'date',stricter = True )
    cgdd_df = gdd_interpol(df, hm , cgdd,bands ,interpol_method = interpol_method,increment = increment,extreme = extreme,  maxcgdd = maxcgdd, date_method = date_method , verbose =verbose)
    return cgdd_df 

if __name__ == '__main__':
    
    import GlobalVars 
    date_method = 'fixed'
    startdate = '2023-01-01' 
    hm = pd.read_csv('Data\Input\harmonisedv5\harmonised_v5_2023_csv\hmv52023.csv')
    which = 'optical' 
    
    if(which=='optical'):
        all_df= pd.read_csv('Data\\Input\\Satellite\\Optical0.1.csv')
        BANDS = GlobalVars.optical_bands
    elif(which=='sar'):
        all_df= pd.read_csv('Data\\Sentinel\\all_sar_restructured_cleaned.csv')
        BANDS = GlobalVars.sar_bands
        
    if(date_method=='fixed'):
        cgdd = pd.read_csv('Data\\interim\\post\\cgdd_v5_fixed_'+startdate+'.csv')
    else:
        cgdd = pd.read_csv('Data\\interim\\post\\cgdd_v5_dynamic.csv')
        
    params = {    
    'verbose' :True  , 
    'interpol_method' : 'closest' , 
    'increment' : 50 ,
    'extreme' : 10 ,  # nulls 15 - 502879  , 10 502944  
    'maxcgdd' : 5000 ,}
    
    all_df = utils.fix_column_syntax(all_df , from_re=r'[0-9]{8}_',impose_date = True )
    all_df_interpoled = pipeline_executable(all_df, hm , cgdd,bands = BANDS , **params)
    all_df_interpoled.to_csv('Data\Interim\cgdd\cgdd_beta002.csv')
    
    
    
'+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+='


