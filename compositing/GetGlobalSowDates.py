# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:03:46 2024

@author: kashy
"""


import pandas as pd
from datetime import datetime
import numpy as np


def mean_datetime(datetimes):
    timestamps = [dt.timestamp() for dt in datetimes]
    mean_timestamp = sum(timestamps) / len(timestamps)
    mean_datetime = datetime.fromtimestamp(mean_timestamp)
    return mean_datetime

def get_datetime(datestr):
    date_format = "%Y-%m-%d"
    if(type(datestr) == float):
        return -1
    date_time_obj = datetime.strptime(datestr, date_format)
    return date_time_obj




def GetGlobalSowDates(df):
    crops = np.unique(df.Crop_Type)

    crop_dates = {}
    for crop in crops:
        print('Crop is ',crop)
        cropdf = df[df.Crop_Type==crop]
        sowingdates = cropdf.Sow_Date.values
        sowingdates = [get_datetime(date) for date in sowingdates]
        sowingdates = [date for date in sowingdates if date != -1]
        if(len(sowingdates)<1):
            crop_dates[crop] = -1
            continue
        meandate = mean_datetime(sowingdates)
        crop_dates[crop] = meandate.strftime('%Y-%m-%d')
    return crop_dates


if __name__ == '__main__':
    
    df = pd.read_csv('Data\\input\\harmonisedv5\\harmonised_v5_2023_csv\\hmv52023.csv')
    crop_dates = GetGlobalSowDates(df)
    print(crop_dates)


    
    