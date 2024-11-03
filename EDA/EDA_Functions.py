# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:33:58 2024

@author: kashy
"""

import matplotlib.pyplot as plt 
import seaborn as sns 

import pandas as pd 
import numpy as np 
import re 





#+++++++++++++++++++++++++   Satellite EDA functions    ++++++++++++++++++++++++++++++++++#                    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_numeric_columns(df):
    numeric_columns = []
    
    for col in df.columns:
        # Check if all values in the column are numeric
        if pd.to_numeric(df[col], errors='coerce').notnull().all():
            numeric_columns.append(col)
    
    return numeric_columns

def get_timesteps_bands(df , reg = -1 , check = False):
    '''
    1. Get timesteps and unique bands (only those with timesteps) that the dataframe has. 
    2. Check that all these bands have all timesteps. (Columns should either have all, or none of the timesteps)
    '''
    if(reg==-1):
        print("Regex wasn't set. Setting this to '[0-9]+__'. Note, this is not imposing YYYYMMDD and you should really send the regex you want. ")
        reg = '[0-9]+__'
        #The regex is numbers of any length followed by double underscore.
     
    
    # int() typecast works because we are using YYYYMMDD and not DDMMYYYY, and there's no leading zeroes.
    # It also works for redone timesteps. 
    timesteps , bands = zip(*[col.split('__') for col in df.columns if re.match(reg,col)])
    timesteps_unique , bands_unique = list(sorted(set(timesteps))) , list(set(bands))
    
    from collections import Counter 
    #A_K_  check this
    if(check):
        tu ,bu = len(timesteps_unique) ,len(bands_unique)
        timesteps_counter , bands_counter = Counter(timesteps) , Counter(bands)
        
        for timestep in timesteps_unique: 
            assert timesteps_counter[timestep] == bu ,"Some timesteps seem to be missing some bands."
        for band in bands_unique:
            assert bands_counter[band] == tu , "Some bands seem to be missing all timesteps."
    
    return timesteps_unique, bands_unique 

def box(df):
  plt.figure(figsize=(10, 6))
  sns.boxplot(data=df)
  plt.xlabel('Column')
  plt.ylabel('Value')
  plt.title('Outliers in Columns')
  plt.show()
  
def plot_histogram(data):
  'DEFUNCT Why do you need seperate graphs with weird dimentions, just call a single one seperately.'
  cols = data.columns
  fig, axs = plt.subplots(1,len(cols))
  for i in range(len(cols)):
    col = cols[i]
    if(len(cols)==1):
      a = axs
    else:
      a = axs[i]
    sns.histplot(data=data, x=col, kde=True, color="skyblue", ax=a)
  plt.tight_layout()


def single_timestep(df,time,other=True):
  '''
  return single timestep from all.
  Change column names , remove <num>_
  Additionally, choose whether you want Other columns (Kernel Data, time independant indices).
  '''

  columns = [col for col in df.columns if(re.match(str(time)+'__',col))] #If you use search here, and time=0, it will match time=10as well.
  
  others = [col for col in df.columns if( not re.search(r'[0-9]_',col))]
  newdf = df[columns]
  newdf.columns = newdf.columns.str.replace(r'^[0-9]+__','',regex=True)
  if(other):
    #newdf.loc[:,others] = df[others]
    newdf = pd.concat([newdf,df[others]],axis=1)
  return newdf

def bar_hist(df , col = 'Crop_Type',method = 'cat',num_bins = 5,h=True):
    '''
    For Categorical, print bar graph. 
    For Real, print histogram. 
    value_counts(df,col='Crop_Type',method = 'cat')
    value_counts(df,col='0__NDVI',method = 'reg',num_bins = 20,h=False)
    
    '''
    if(method ==  'cat'):
        df[col].value_counts().plot(kind='barh')
    elif(method =='reg'):

        hist_values, bin_edges = np.histogram(df[col], bins=num_bins) 
        plt.figure(figsize=(10, 6))
        if(h):
            plt.barh(bin_edges[:-1], hist_values, height=np.diff(bin_edges), color='skyblue', edgecolor='black', align='edge')
            plt.xlabel('Frequency')
            plt.ylabel('Values')
        else:
            plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), edgecolor='black', align='edge')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            

        plt.title('Histogram of column_name')

    else:
        1/0

def merge_time(df,timesteps = []):
  '''
  merge all times into a single dataframe.
  Think about other columns
  '''
  # A_K_
  
  3/0,'edit'

  lis = []
  for time in timesteps:
    lis.append(single_timestep(df,time=time))
  return pd.concat(lis)


# Helpers - 
## 1. Get numeric columns 
## 2. Get base columns -> COPY OF get_timesteps_bands (will be sorted)
## 3. Boxplot
## 4. DEFUCNT (use 6).Plot_histogram 
## 5. single timestep. 
## 6. bar_hist - bar for categoriical, hist for real. 
## 7. Merge time


" Don't include special columns at first."
" Maybe for points and time, there could be single, multiple and multiple aggregated as single"


# -> A. for all of 4th dim  B. For 1 specific 4th dim C. For 
## this 4th dim can be croptype , Sowdate existance , Location, etc. It is alway categorical. 
## 4th dim categorical can be hue for any scatterplot.

# 1. Single Aggregations

## 1.1 Same time, Same bands, Multiple points
#### Describe column, Value counts. Look for meaningful time-band pairs.
#### Scatter plot, Variance of the column. wrt target, see correlation with target. 
#### Histogram of a band , skewness of columns. 
def skew_division(df,cols):
  imp = df[cols]
  low_skew = []
  med_skew = []
  high_skew = []
  for col in imp:
      skew = imp[col].skew()
      if(abs(skew)>1):
        high_skew.append((col,skew))
      elif(abs(skew)>0.5):
        med_skew.append((col,skew))
      else:
        low_skew.append((col,skew))
  print("No of columns with : \n1. Low Skew :{}\n2. Med Skew :{} \n3. High Skew :{} ".format(len(low_skew),len(med_skew),len(high_skew)))
  low_skew.sort(key = lambda x: x[1])  , med_skew.sort(key = lambda x: x[1]) , high_skew.sort(key = lambda x: x[1])
  return low_skew , med_skew , high_skew



## 1.2 Same time, Multiple bands, 1 point 
#### Sample bar chart of a point's signature. 
 



## 1.3 Multiple time , Same band , 1 point
####  Sample timeseries of a point. 



# 2. Double Aggregations

## 2.1 One time , Multiple Bands , Multiple points 
####  Bar chart with variances of each band. 
#### (Future) With max, min, mean etc of each band, bar chart.
#### Boxplot of a timestep (A timestep's signature.)
#### Bar chart of the skewness of bands in a single time.
def plot_skewness(data,figsize=(10,6)):
  skewness = data.skew()
  #skewness = skewness.dropna()
  skewness.plot(kind='bar', figsize=figsize)
  plt.xlabel('Column')
  plt.ylabel('Skewness')
  plt.title('Skewness of Columns')
  plt.show()

def get_col_variances(data,threshold=0):
    variances = data.var().sort_values(ascending=False)
    retval = []
    for column, variance in variances.iteritems():
      if(variance>threshold):
        retval.append((column,variance))
    return retval


## 2.2 Multiple times , Same band , Multiple Points 
####  Time series with mean, variance (or) median , variance. 
#### Box plot of a band (A band's signature)
    
def plot_mean_std(df,bands = ['NDVI'],croptypes = [],method='mean',show=True,std=True,append='',xl='GDD',tle='Mean and std for band'):
    '''
    Multiple Timesteps, Single band, Multiple points. 
    method: bands, croptype
    '''
    sns.set(style="darkgrid")
    
    
    assert len(croptypes)!=0, "Croptypes should be non zero"
    for croptype in croptypes:
        df_crop = df[df.Crop_Type == croptype]
        for band in bands:
            if(method=='mean'):
                dfmean = df_crop[[col for col in df_crop.columns if(band in col and re.match(r'[0-9]+__',col))]].mean(axis=0)
            elif(method=='median'):
                dfmean = df_crop[[col for col in df_crop.columns if(band in col and re.match(r'[0-9]+__',col) )]].median(axis=0)
            else:
                7/0, 'method not recognized'
            values = sorted(list(set([int(col.split('__')[0])  for col in dfmean.index])))
            values = [ dfmean[str(val)+'__'+band] for val in values]
            sns.lineplot(x=range(len(values)),y=values,label=f'{croptype} - {band} {append}')

            
            dfstd = df[[col for col in df_crop.columns if(band in col and re.match(r'[0-9]+__',col))]].std(axis=0)
            stdvalues = sorted(list(set([int(col.split('__')[0]) for col in dfstd.index ])))
            stdvalues = [ dfstd[str(val)+'__'+band] for val in  stdvalues]
            
            
            plt.fill_between(range(len(stdvalues)), np.array(values) - np.array(stdvalues), np.array(values) + np.array(stdvalues), color='blue', alpha=0.2) if std==True else -1

    if(show):
        plt.xlabel('Time / GDD axis. ')
        plt.ylabel(f'Bands - {bands}')
        plt.title('Mean band series')
    return values, stdvalues

#How time series of the average value of a band across all indices compare?
def band_series_by_croptype(df,band,crops = []):
  '''
  If crops is null, get all crops.
  Else, get specified crops
  '''
  def resort(cols):
    dig = sorted([int(col.split('__')[0]) for col in cols])
    z = list(set([col.split('__')[1] for col in cols]))
    assert len(z) == 1
    return [ str(di)+'__'+z[0] for di in dig]
  lines = []
  legends = []
  plt.figure(figsize=(20,5))
  if(len(crops) == 0):
    crops = np.unique(df.Crop_Type)
  for Crop_Type in crops:
    crop = df[df.Crop_Type == Crop_Type]
    #line = [crop[col].mean() for col in crop.columns if(re.search(r'^[0-9]{1,2}_'+band+'$',col))] #End operator, $ is important here.
    cols = resort([col for col in crop.columns if(re.search(r'^[0-9]{1,2}__'+band+'$',col))]) # want 10 to be after 9, not after 1
    line = [crop[col].mean() for col in cols]
    lines.append(line)
    legends.append(Crop_Type)
    plt.plot(line,label = Crop_Type)
  plt.title(f'Average values over time for: {band}')
  plt.legend()
  plt.grid()
  plt.show()
  plt.figure(figsize = (20,5))
  x = range(len(lines[0]))
  plt.stackplot(x,np.array(lines),labels=legends)
  plt.legend(loc='upper left')


## 2.3 Multiple times , Multiple bands , Same point. 
#### Time series signature of a point. 



#+++++++++++++++++++++++++  Others    ++++++++++++++++++++++++++++++++++#                    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def value_counts(df , col = 'Crop_Type',method = 'cat'):
    if(method ==  'cat'):
        df[col].value_counts().plot(kind='barh')
    elif(method =='reg'):
        
        num_bins = 5
        hist_values, bin_edges = np.histogram(df[col], bins=num_bins) 
        plt.figure(figsize=(10, 6))
        plt.barh(bin_edges[:-1], hist_values, height=np.diff(bin_edges), color='skyblue', edgecolor='black', align='edge')

    else:
        1/0




