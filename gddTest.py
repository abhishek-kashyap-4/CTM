# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 04:10:28 2024

@author: kashy
"""
import pandas as pd 
from utils import utils 

import compositing.temp as gdd 
import compositing.Harmonised_Time_Composite as time

import GlobalVars 

##########################################################
hm_temperatures  = pd.read_csv("Data\\Interim\\post\\hm_v5_2023_temperatures.csv",index_col=0)
hm = pd.read_csv('Data\Input\harmonisedv5\harmonised_v5_2023_csv\hmv52023.csv')
df = pd.read_csv('Data\\Input\\Satellite\\Field_Optical.csv')
df  = df.iloc[210:211]

df = df[df.shape[0]//2:]
df = utils.fix_column_syntax(df,from_re = r'[0-9]+_')


################################################################
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','CLDPRB']

fixed = False 
increment = 20
fixed_date = '20230101'


dftime = time.pipeline_executable(df,BANDS,fixed = fixed , fixed_date = fixed_date,increment = increment)


dfgdd , anchors = gdd.CGDD(df,hm_temperatures , hm , increment = 50)


##############################################################
import matplotlib.pyplot as plt 

cols1 = [col for col in df.columns if 'B3' in col]
cols2 = [col for col in dftime.columns if 'B3' in col]
cols3 = [col for col in dfgdd.columns if 'B3' in col]

v1 = df[cols1].iloc[0].values 
v2 = dftime[cols2].iloc[0].values 
v3 = dfgdd[cols3].iloc[0].values



plt.figure(1)
plt.scatter(x = range(len(v2)) , y= v2)
plt.scatter(x = range(len(v3)) , y = v3 , color = 'red')
plt.show()



