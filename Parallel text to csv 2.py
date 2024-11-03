# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:37:15 2024

@author: kashy
"""

'''
The text consists of seperate lines with geometry # bands as key value pairs.
the geometry is a centroid. Using sjoin for this.

'''

#Combining KNM or EnR files gives you a file too big. do i individually.

'''
This file has been reclaimed 
for EnR. 
'''

import pandas as pd 
from shapely.geometry import Point, Polygon , shape
import json 

import geopandas as gpd
import ast


 


def convert_sysindex_to_time_keys(d,kind):
    '''
    d has rows made up of sysindexes. 
    for optical and sar, extracting timestamp is different, 
    '''
    if(kind=='SAR'):
        raise NotImplementedError
    elif(kind=='Optical'):
        newd = {key[:8]+'_'+key.split('_')[-1]:d[key] for key in d}
    else:
        raise Exception(f"Bad value for kind -{kind} ")
    return newd


def get_csv(rootfilename = 'EnR'):
    '''
    This function was updated to reflect the EnR version of the data. 
    In this, individual subsections of presampled datapoints 
        were put in a multiprocessing colab environ to fetch satellite data.
    Data was fetched line geom#{dictionary of sysindex_band: value }
    Every subsection had a skipfile which contained the indexes that returned null. 
    HOWEVER, these aren't true indexes as per the original sampled dataframe, 
    but rather count started from the first point that the subsection contained. 
    So, If the subsection was 30000 to  35000 ,  in the skipfile, number 300000 corresponds to 0. 
    So, you always have to add the respective starting number, which is included in the filename. 
    
    
    This function gets all individual files, and return a merged dataframe. 
    The structure of the file is expected to be like
    <root_file_name>#from-to.txt
    
    !! Important !! rootfilename should also be the directory name. 
    example - Data/EnR/EnR#1000-2000.txt
    
    '''
    
    fnames = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000-30000',
              '30000-35000','35000-40000','40000-45000','45000-50000','50000-end']
    fnames.remove('40000-45000') #for now this is absent 
    
    skipfileadds = [int(val.split('-')[0]) for val in fnames]
    
    rows = []
    skipindexes = []
    for fname,skipfileadd in zip(fnames,skipfileadds): 
        with open('Data/'+rootfilename+'/'+rootfilename+'#'+fname+'.txt','r') as f:
            r = f.readlines()
            count = len(r)
        for line in r:
            geom  = line.split("#")[0].split('_')
            d = json.loads(line[:-1].split('#')[1])
            d = convert_sysindex_to_time_keys(d,kind='Optical')
            rows.append(d | {'.geo':geom})
        
        with open('Data/'+rootfilename+'/'+'skipfile add '+str(skipfileadd)+'.txt','r') as f:
            r1 = f.readlines() 
        skipindexes += [int(line) + skipfileadd for line in r1]
    
    rows = pd.DataFrame(rows)
    return rows  , pd.Series(skipindexes)

def remove_nulls(df):
    df_cleaned = df.dropna(how='all', subset=df.columns.difference(['.geo']))
    print( (len(df)-len(df_cleaned))/len(df))
    return df_cleaned



hm = pd.read_csv('Data\\input\\harmonisedv5\\harmonised_v5_2023_csv\\hmv52023.csv') 
hm['.geo'] = hm['.geo'].apply(lambda x: shape(json.loads(x)))
gdf_hm = gpd.GeoDataFrame(hm, geometry='.geo')



#df,skipindexes = get_csv()
#df.to_csv('Data/EnR.csv')
#pd.DataFrame(skipindexes).to_csv('Data/skipsfull.csv')


df = pd.read_csv('Data/EnR.csv').drop(columns = ['Unnamed: 0'])
df = remove_nulls(df)

df['.geo'] = df['.geo'].apply(lambda x: Point(list(ast.literal_eval(x))))
gdf_df = gpd.GeoDataFrame(df, geometry='.geo')

joined_gdf = gpd.sjoin(gdf_df, gdf_hm, how='left', predicate='within')
joined_gdf = joined_gdf.drop(columns=['index_right'])

joined_gdf.to_csv('Data\\input\\Satellite\\Optical.csv')





    



















