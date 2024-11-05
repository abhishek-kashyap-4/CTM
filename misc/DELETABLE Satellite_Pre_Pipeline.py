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

def stat_analysis(df,plot=True):
    '''
    Given a df with time or gdd axis, get stats such as 
    Mean, STD.

    '''   
def plot_mean_std(df,band='NDVI',std=True,fig=0,xl='GDD',clr='blue',tle ='Mean and Std for band'):
    dfmean = df[[col for col in df.columns if(band in col )]].mean(axis=0)
    dfstd = df[[col for col in df.columns if(band in col )]].std(axis=0)
    plt.figure(fig)
    values = [ dfmean[str(val)+'_'+band] for val in range(len(dfmean))]
    plt.plot(range(len(values)),values,color='blue')
    plt.fill_between(range(len(values)), dfmean - dfstd, dfmean + dfstd, color=clr, alpha=0.2) if std==True else -1
    plt.xlabel(xl)
    plt.ylabel(band)
    plt.title(tle)
    plt.legend()
    plt.show()
    
def plot_means(df,bands = ['NDVI'],croptypes = [],method='mean',show=True,std=True,append='',xl='GDD',tle='Mean and std for band'):
    '''
    method: bands, croptype
    '''
    sns.set(style="darkgrid")
    
    
    assert len(croptypes)!=0, "Croptypes should be non zero"
    for croptype in croptypes:
        df_crop = df[df.Crop_Type == croptype]
        for band in bands:
            if(method=='mean'):
                dfmean = df_crop[[col for col in df_crop.columns if(band in col )]].mean(axis=0)
            elif(method=='median'):
                dfmean = df_crop[[col for col in df_crop.columns if(band in col )]].median(axis=0)
            else:
                7/0, 'method not recognized'
            values = [ dfmean[str(val)+'_'+band] for val in range(len(dfmean))]
            sns.lineplot(x=range(len(values)),y=values,label=f'{croptype} - {band} {append}')
            dfstd = df[[col for col in df.columns if(band in col )]].std(axis=0)
            plt.fill_between(range(len(values)), dfmean - dfstd, dfmean + dfstd, color='blue', alpha=0.2) if std==True else -1

    if(show):
        plt.xlabel('GDD')
        plt.ylabel('NDVI')
        plt.title('Mean NDVI series')
                    

        
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
    other_stupid_verbose['gddrange'] = gddrange          
    for gdd in gddrange: 
        if( str(gdd) in row and not math.isnan(row[str(gdd)])):
            cols.append(str(gdd)+'_DATE')
            gdds.append(str(gdd))    
    return row[cols] , row[gdds]
# %%

"##### Level 2 functions ##"
def intepol_singleline(row ,index,date,method='lookbehind',n=3,extreme=20):
    '''
    USING THE DATES ARE SORTED.
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
    global gdd_interpol_verbose #Use this to store variables you want to see in explorer.
    if(method in ('reduce_mean_n','reduce_median_n')):
        raise NotImplementedError
    
    def get_left_right(datetouse,extreme):
        lefts = []
        rights = []
        for i in range(1,extreme):
            left = datetouse - timedelta(days=i)
            left = left.strftime('%Y%m%d')
            if(left+'_'+BANDS[0] in row):
                s = sum([math.isnan(row[left+'_'+band]) for band in BANDS]) 
                if(s==0):
                    lefts.append(left)
                else:
                    assert  s==len(BANDS), f"Null value inconsistency with S being {s}"
            right = datetouse + timedelta(days=i)
            right = right.strftime('%Y%m%d')
            if(right+'_'+BANDS[0] in row):
                s = sum([math.isnan(row[right+'_'+band]) for band in BANDS]) 
                if(s==0):
                    rights.append(right)
                else:
                    assert  s==len(BANDS), f"Null value inconsistency with S being {s}"
        return lefts,rights
    def date_validity(row,datetouse):
        if(datetouse+'_'+BANDS[0] in row ):
            s = sum([math.isnan(row[datetouse+'_'+band]) for band in BANDS]) 
            if(s==0):
                return True
            assert s == len(BANDS)
        return False
        
    #####
    datetouse_date = datetime.strptime(str(int(date)) , '%Y%m%d')
    datetouse = datetouse_date.strftime('%Y%m%d')
    
    
    if(method in ('lookbehind','lookahead','closest')):
        lefts,rights = get_left_right(datetouse_date,extreme =2) #You should give 2 for one value.
        if(datetouse+'_'+BANDS[0] in row ):
            if(date_validity(row,datetouse)):
                return {str(index)+'_'+band : row[datetouse+'_'+band] for band in BANDS}
        if(method =='lookbehind'):
            assert len(lefts) !=0 , f"For extreme {extreme}, lefts is zero for date, {date}"
            return {str(index)+'_'+band : row[lefts[-1]+'_'+band] for band in BANDS}
        elif(method=='lookahead'):
            return {str(index)+'_'+band : row[rights[0]+'_'+band] for band in BANDS}
        elif(method=='closest'):
            #Heres the thing, you can just send closest date,
            #and that will be easy, but technically speaking
            #we want the closest date to the correct CGDD, 
            #Which isn't the same as closest to datetouse.
            #You need a cgdd of not just 25 increments, but everything. 
            if(len(lefts)==0 and len(rights)==0):
                return {} #What can men do against such reckless hate?
            if(len(lefts)==0):
                return {str(index)+'_'+band : row[rights[0]+'_'+band] for band in BANDS}
            if(len(rights)==0):
                return {str(index)+'_'+band : row[lefts[-1]+'_'+band] for band in BANDS}
            
            if(datetouse_date - datetime.strptime(lefts[-1], '%Y%m%d') < \
               datetime.strptime(rights[0], '%Y%m%d')-datetouse_date):
                
                return {str(index)+'_'+band : row[lefts[-1]+'_'+band] for band in BANDS}
            else:
                return {str(index)+'_'+band : row[rights[0]+'_'+band] for band in BANDS}

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
            for band in BANDS:
                retval[str(index)+'_'+band] = np.array([row[date+'_'+band] for date in dates]).mean()
        elif('median' in method):
            for band in BANDS:
                retval[str(index)+'_'+band] = np.median(np.array([row[date+'_'+band] for date in dates]))
        else:
            36/0 , 
        return retval
            
                
def gdd_interpol(all_df, hm , cgdd):
    '''
    Given the 3 dataframes, return df with cgdd axis.
    Neither of all_df , cgdd need to have all the points. So loop with hm.
    Dependancies of the function are all in gdd_dependancies.
    '''
    global gdd_interpol_verbose #Use this to store variables you want to see in explorer.
    COLUMNS  = [col for col in all_df.columns if re.match(r'[0-9]',col)]
    unique_ids = np.unique(hm.Field_Id)
    newrows = []
    for fid in unique_ids:
        print(gdd_interpol_verbose['progress']/len(unique_ids)) if verbose else -1
        gdd_interpol_verbose['progress'] += 1
        hmrow = hm[hm.Field_Id==fid].iloc[0]
        #Perhaps, it needs to be checked.
        if(fid not in cgdd['Field_Id'].values):
            continue 
        if(fid not in all_df['Field_Id'].values):
            continue
        if(fid in gdd_dependancies['exceptions2'] or \
           fid in  gdd_dependancies['red_geom']['index'] ):
            raise Exception("FIDs need to be V4.")
            continue
        #I am not writing checks for -
            # If not in cgdd it must be fallow or abandoned
            # If not in restructured_cleaned, it must redundant or non polygon 
        #Because this code should work for parial data as well.
        datesdf = cgdd[cgdd['Field_Id']==fid]
        sardf = all_optical[all_optical['Field_Id']==fid]
        #sardf = aggregate_bands(sardf,COLUMNS)
        assert len(datesdf)==1  , f'FID {fid},datesdf size is {len(datesdf)}'
        assert len(sardf) in (1,2) , (fid , len(sardf))
        dates ,gdds= getdates(datesdf.iloc[0],maxcgdd=gdd_dependancies['maxcgdd'],increment=gdd_dependancies['increment'])
        
        newrow = {}
        for rownum,row in sardf.iterrows():
            gdd_interpol_verbose['sanity'] += 1
            for index,date in enumerate(dates):
                newrow |= intepol_singleline(row ,index,date,method= interpol_method,extreme=gdd_dependancies['extreme'])
            newrow['Crop_Type'] = hmrow['Crop_Type']
            newrow['Field_Id'] = hmrow['Field_Id']
            if(date_method=='fixed'):
                newrow['Sow_Date'] = startdate_fixed
            else:
                newrow['Sow_Date'] = hmrow['Sow_Date']
        newrows.append(newrow)
    return pd.DataFrame(newrows)
            


# %%
#########################################################################
from  Project.feature_add import feature_addition

"+=+=+=+=+=+=+=+=+=+=+==GLOBAL VARIABLES+=+=+=+=+=+=+=+=+=+=+="
which = 'optical'
verbose = True 
date_method='fixedbleh'
interpol_method = 'closest'
interpol_method_str = ''.join(word.capitalize() for word in interpol_method.split('_'))
savefiles = True
print("Change method options in gdd_singleline to camel case.")
gdd_dependancies = { #This data is used in the gdd_interpol() function. 
     'red_geom' : pd.read_csv('Data\\post\\redundant_geometries_with_fid.csv'),
     'exceptions2':[2519, 2522, 2541, 2555, 2561, 2566, 2746, 2969, 3052, 3201, 3208],
     #Exceptions is a list of field Ids that I am ignoring. 
     #These fields have various problems. But these Ids are of V3. I need V4.
    'exceptions3': [2165] ,
     #Fields such as this are incorrect, since they are summer crops yet the sowing date
     #Is in winter. They have to be removed.
    'maxcgdd' : 4225 if date_method=='fixed' else 5450,  
    #This is manual here, but you can refer to cgdd for this.
    #Why had maxcgdd changed from v3? Probable the 6200 thing had been removed...
    #btw, for startdate 10 01, (from where i have Optical data, maxdgdd is 4775)
    'increment' : 50,
    'extreme' : 5
    }
gdd_interpol_verbose = {
    'absolutetotalskips' : 0,
    'sanity' : 0,
    'progress' : 1} 
other_stupid_verbose = {}
allc  =['Grassland' , 'Maize','Maize_Silage','Oats','Spring_Wheat' ,
'Sugar_Beet','Sunflower','Wheat','Winter_Barley','Winter_Rapeseed',
'Winter_Rye','Winter_Wheat']
allc = ['Grassland' , 'Maize','Spring_Wheat' ,
'Sunflower','Wheat','Winter_Rapeseed',
'Winter_Wheat']
winter = ['Wheat','Winter_Barley','Winter_Rapeseed',
'Winter_Rye','Winter_Wheat']

if(which=='optical'):
    all_optical= pd.read_csv('Data\\Sentinel\\all_optical_restructured_cleaned.csv')
    #2023 for sar and 2024 for optical, for now. 
    hm = pd.read_csv('Data\\harmonisedv4\\harmonised_v4_2023_csv\\hmv42023.csv')
    cgdd = pd.read_csv('Data\\post\\cgdd_v4.csv')
    BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','MSK_CLDPRB']

elif(which=='sar'):
    all_sar= pd.read_csv('Data\\Sentinel\\all_sar_restructured_cleaned.csv')
    #2023 for sar and 2024 for optical, for now. 
    hm = pd.read_csv('Data\\harmonisedv3\\harmonised_2023_csv\\hm2023.csv')
    cgdd = pd.read_csv('Data\\post\\cgdd_v3.csv')
    exceptions2 = [2519, 2522, 2541, 2555, 2561, 2566, 2746, 2969, 3052, 3201, 3208]
    BANDS = ['VV','VH','VV_VH']

else:
    12/0 

if(date_method=='fixed'):
    assert which=='optical', "Fixed method not available for SAR data at this time."
    cgdd = pd.read_csv('Data\\post\\cgdd_v4_fixed_jan1.csv')
    startdate_fixed = '2023-01-01'
'+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+='
if(verbose):
    print("""!!!!! VERY IMPORTANT.
        Your CGDD stil has redundat fields and weird geometries. 
        Your all_restructured_cleaned still has fallow and abandoned with are abset in cgdd.
        """) 

    print("Problem - the data (time) is insufficient.")
    print("But this would just work for the winter crops.")
    
# %% 
if __name__== '__main__':
    if(True):
        ######################### WORKING ################################################
        
        all_optical_interpoled = gdd_interpol(all_optical, hm , cgdd)
        featured = True 
        if(featured):
            timesteps = max(int(val.split('_')[0]) for val in all_optical_interpoled.columns if('B3' in val))
            all_optical_interpoled = feature_addition(all_optical_interpoled.copy(), range(timesteps))
        savefiles = False
        if(savefiles):       
            if(date_method=='fixed'):
                if(featured):
                    fname = f'Project\\Satellite\\Optical_GDD_Method{interpol_method_str}_fixedStart_jan1_featured.csv'
                else:
                    fname = f'Project\\Satellite\\Optical_GDD_Method{interpol_method_str}_fixedStart_jan1.csv'
                
            else:
                if(featured):
                    fname = f'Project\\Satellite\\Optical_GDD_Method{interpol_method_str}_featured.csv'
                else:
                    fname = f'Project\\Satellite\\Optical_GDD_Method{interpol_method_str}.csv'
            
            all_optical_interpoled .to_csv(fname)  
        
        plots = True 
        if(plots):
            
            dftime_all = pd.read_csv('Project\\Satellite\\all_optical_10day_featured.csv').drop(columns = ['Unnamed: 0'])
            COLUMNS = [col for col in dftime_all.columns if re.match(r'[0-9]',col)]
            COLUMNS += ['Crop_Type']
            dftime_all = dftime_all[COLUMNS]
            ctypes = ['Winter_Wheat','Winter_Rapeseed']
            plt.figure(1)
            plot_means(all_optical_interpoled,bands = ['NDVI'],croptypes = ctypes,method='mean',\
                   show=True,append = 'increment_'+str(gdd_dependancies['increment']),std=True)
            plt.figure(30)
            plot_means(dftime_all,bands = ['NDVI'],croptypes = ctypes,method='mean',\
                   show=True,std=True)
            
        1/0
        plot_means(all_optical_interpoled,bands = ['NDVI'],croptypes = winter,method='mean',\
               xl='GDD',tle='Mean and std for band')
    
    # %%
    
    #allc = np.unique(all_optical_interpoled.Crop_Type)
    ctypes = ['Winter_Rapeseed','Wheat','Maize','Sunflower','Grassland']
    ctypes = ['Wheat']
    all_optical = all_optical[all_optical.Crop_Type.isin(ctypes)]
    
    dftime_all = pd.read_csv('Project\\Satellite\\all_optical_15day_featured.csv').drop(columns = ['Unnamed: 0'])
    COLUMNS = [col for col in dftime_all.columns if re.match(r'[0-9]',col)]
    COLUMNS += ['Crop_Type']
    dftime_all = dftime_all[COLUMNS]
    wheat_time = dftime_all[dftime_all.Crop_Type=='Wheat']
    
    wheat_NDVI_time = wheat_time[[col for col in wheat_time.columns if 'NDVI'  in col]].mean()
    
    dfs = []
    for i in range(75,251,25):
        gdd_dependancies['increment'] = i
        all_optical_interpoled = gdd_interpol(all_optical, hm , cgdd)
        timesteps = max(int(val.split('_')[0]) for val in all_optical_interpoled.columns if('B3' in val))
        all_optical_interpoled = feature_addition(all_optical_interpoled.copy(), range(timesteps))
        COLS = [col for col in all_optical_interpoled.columns if 'NDVI' in col or 'Crop_Type' in col]
        all_optical_interpoled = all_optical_interpoled[COLS]
        dfs.append(all_optical_interpoled)
    
    for i in range(len(dfs)):
        df = dfs[i]
        plt.figure(i)
        plot_means(df,bands = ['NDVI'],croptypes = ctypes,method='mean',\
               show=True,append = 'increment_'+str((i+1)*75),std=True)
        
    plt.legend()
    plt.show()
    
    plt.figure(30)
    plot_means(dftime_all,bands = ['NDVI'],croptypes = ctypes,method='mean',\
           show=True,std=True)
    
    
    '''plt.figure(-1)
    plot_means(dftime_all,bands = ['NDVI'],\
               croptypes = ['Winter_Rapeseed','Wheat','Maize','Sunflower','Grassland'],\
                   method='mean',show=False,std=False)
    plt.xlabel('Time')
    plt.ylabel('NDVI')
    plt.title('NDVI Time series')
    plt.show()
    
    
    COLUMNS = [col for col in all_optical.columns if re.match(r'[0-9]',col)]
    COLUMNS += ['Crop_Type']
    croptypes = all_optical[COLUMNS].groupby('Crop_Type').mean().plot()
    '''
    
    
    #croptypes = all_optical[COLUMNS].groupby('Crop_Type').mean().plot()
    
    
    
    
    
    
    
    
