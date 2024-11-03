# -*- coding: utf-8 -*-
"""
Created on Tue Jul  15 00:21:58 2024

@author: kashy
"""

'''
Reshaping all.csv of either SAR or Optical to proper format, 
correcting v3 error by removing redundant geometries.
'''
import numpy as np
import pandas as pd 
from datetime import datetime,timedelta
import math

#INPUTS
"####centroid_V3_2023_temperatures.csv"
"This has centroids with dates for 25C increment for GDD. It also has Fid."
"####all_Optical_restructured_cleaned.csv"
"This has the actual satellite date rowwise. it has fid aussi"

print("""!!!!! VERY IMPORTANT.
    Your CGDD stil has redundat fields and weird geometries. 
    Your all_restructured_cleaned still has fallow and abandoned with are abset in cgdd.
    """)

print("Problem - the data (time) is insufficient.")
print("But this would just work for the winter crops.")

print("You've incorrectly dismissed zeroth column from CGDD. It should be included.")

print("croptype from hm isn't required anymore as the restructured_cleaned now has croptype")
all_optical= pd.read_csv('Data\\Sentinel\\all_optical_restructured_cleaned.csv')
hm2023 = pd.read_csv('Data\\harmonisedv4\\harmonised_v4_2023_csv\\hmv42023.csv')
cgdd = pd.read_csv('Data\\post\\cgdd_v4.csv')


#4m small field with geodesc false and different points sameples in every timestep.
exceptions = [2519,  2522,2541,2566,2746,2969]
exceptions2 = [2519, 2522, 2541, 2555, 2561, 2566, 2746, 2969, 3052, 3201, 3208]

maxcgdd = 6200
increment = 100
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','MSK_CLDPRB']

def getdates(row):
    cols = []
    gdds = [] 
    for gdd in range(0+increment,maxcgdd+1,increment): #1st zero is absent, so doing 0+ increment
        if(str(gdd) not in row):
            return row[cols] ,row[gdds]
        if(not math.isnan(row[str(gdd)])):
            cols.append(str(gdd)+'_DATE')
            gdds.append(str(gdd))    
    return row[cols] , row[gdds]

def many_asserts(row,d):
    'if the date d exists, either all bands need to be null or none.'
    if(d+'_'+BANDS[0] in row):
        s=sum([math.isnan(row[d+'_'+band]) for band in BANDS])
        if(s ==0):
           return True  # We found the date with no nulls. 
        assert  s==len(BANDS), "Null value inconsistency."
            
    return False #Don't break since we didn't find the right date

absolutetotalskips = 0
sanity = 0
progress = 1
newrows = []
fids = []
'''
for fid in np.unique(hm2023.Field_Id):
    hmrow = hm2023[hm2023.Field_Id==fid]
    assert sum(val!=hmrow.Field_Id.iloc[0] for val in hmrow.Field_Id)==0
    if(fid  in all_optical['Field_Id'].values): # For partial data also, this works 
        fids.append(fid)
ctps = [row['Crop_Type'] for rownum,row in hm2023.iterrows() if(row.Field_Id in fids)]

print(pd.Series(ctps).value_counts())
ctps2 = [row['Crop_Type'] for rownum,row in all_optical.iterrows() if(row.Field_Id in fids)]
print(pd.Series(ctps2).value_counts())
1/0
'''
for fid in np.unique(hm2023.Field_Id):
    print(progress/len(np.unique(hm2023.Field_Id)))
    progress += 1
    hmrow = hm2023[hm2023.Field_Id==fid].iloc[0]
    if(fid not in cgdd['Field_Id'].values):
        continue 
    if(fid not in all_optical['Field_Id'].values): # For partial data also, this works 
        continue
    if(fid in exceptions or fid in exceptions2):
        continue
    #Need to write checks for the above 2 things. They should be ignored.
    #if not in cgdd, it must be fallow or abandoned. 
    #if not in restructured_cleaned , it must be redundant fields and non polygon  geometries.
    datesdf = cgdd[cgdd['Field_Id']==fid]
    sardf = all_optical[all_optical['Field_Id']==fid]
    
    assert len(datesdf)==1  , (fid,len(datesdf))
    assert len(sardf) in (1,2) , (fid , len(sardf))
    dates ,gdds= getdates(datesdf.iloc[0])
    newrow = {}
    for rownum,row in sardf.iterrows():
        sanity += 1
        for index,date in enumerate(dates):
            extreme = 20
            datetouse = datetime.strptime(str(int(date)) , '%Y%m%d') 
            while(True):
                d = datetouse.strftime('%Y%m%d')
                breakit = many_asserts(row,d)
                if(breakit):
                    break
                if(extreme<0):
                    absolutetotalskips += 1
                    extreme = -2 
                    break
                extreme -= 1
                datetouse -= timedelta(days = 1)
            if(extreme == -2):
                continue
            
                
            datetouse = datetouse.strftime('%Y%m%d')
            
            for band in BANDS:
                newrow[str(index) + '_' + band] = \
                    row[datetouse+'_'+band]
        newrow['Crop_Type'] = hmrow['Crop_Type']
        newrow['Field_Id'] = hmrow['Field_Id']
        newrow['Sow_Date'] = hmrow['Sow_Date']
        newrows.append(newrow)
            
        
  
newrows = pd.DataFrame(newrows)
newrows.to_csv('Data\\Sentinel\\Optical_v4_GDD_scaled.csv')    

    
    