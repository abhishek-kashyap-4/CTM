# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:04:06 2024

@author: kashy
"""

year = '2023'


# Files
harmonised_file = 'Data/input/harmonisedv5/harmonised_v5_'+year+'_csv/hmv5'+year+'.csv'
optical_file = 'Data/input/Satellite/Optical0.1.csv'
CentroidTemperatures_file = 'Data/interim/post/temperature_V5_'+year+'_centroids.csv'
Cloud_file = 'Data/interim/Cloud/Optical_Cloudfilled.csv'


optical_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','CLDPRB']
sar_bands = ["VV","VH","VV_VH"]

target_remap = {'Abandoned': 'Others','Fallow':'Others' , 'Grassland':'Others',
                'Maize':'Maize','Maize_Silage':'Maize','Mined':'Others',
                'Oats':'Oats','Rapeseed':'Rapeseed','Soybean':'Soybean',
                'Spring_Wheat':'Wheat','Sugar_Beet':'Sugar_Beet','Sunflower':'Sunflower',
                'Wheat':'Wheat','Winter_Barley':'Wheat','Winter_Rapeseed':'Rapeseed','Winter_Rye':'Wheat',
                'Winter_Wheat':'Wheat'}

crop_base_temp = {'Abandoned': 0, 'Fallow': 0, 'Grassland': 0, 
                  'Maize': 10, 'Maize_Silage': 0, 'Mined': 0, 
                  'Oats': 0, 'Rapeseed': 5,'Soybean': 10, 
                  'Spring_Wheat': 4, 'Sugar_Beet': 3.5, 'Sunflower': 7, 
                  'Wheat': 0, 'Winter_Barley': 0,
                  'Winter_Rapeseed': 0, 'Winter_Rye': 0,
                  'Winter_Wheat': 0}

crop_avg_sow_date = {'Abandoned': -1, 'Fallow': -1, 'Grassland': '2023-03-27', 
                     'Maize': '2023-05-05', 'Maize_Silage': '2023-05-14', 
                     'Mined': -1, 'Oats': '2023-03-19', 
                     'Rapeseed': '2022-08-11', 'Soybean': '2023-05-12', 
                     'Spring_Wheat': '2023-03-29', 'Sugar_Beet': '2023-05-04', 
                     'Sunflower': '2023-05-02', 'Winter_Barley': '2022-09-29', 
                     'Winter_Rapeseed': '2022-08-20', 'Winter_Rye': '2022-10-25',
                     'Winter_Wheat': '2022-10-03'}
                     
