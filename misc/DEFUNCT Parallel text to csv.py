# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 05:10:12 2024

@author: kashy
"""

'''
The text consists of seperate lines with geometry # bands as key value pairs.
the geometry is a centroid. You need to write a code to see which original hm it belongs to. 
'''
'''
This isn't the correct way of doing it. 
You have sampling points with the same order. 
don't combine KNM but rather take them individually and susect the hm.
'''

import pandas as pd 
from shapely.geometry import Point, Polygon , shape
import json 
 
hm = pd.read_csv('Data\\input\\harmonisedv5\\harmonised_v5_2023_csv\\hmv52023.csv') 


def inside_geom(geometry,polygon):
    
     '''
     Return True is geometry is in the polygon
     '''
     polygon = json.loads(polygon)
     polygon = shape(polygon)
     
     #geometry = json.loads(geometry)
     geometry = Point(geometry)
     
     return polygon.contains(geometry)
 

def check_respective_hm(geometry,hm):    
    count = 0
    for polygons in hm['.geo']:
        if(inside_geom(geometry,polygons)):
            count += 1
    assert count ==1 


def get_respective_hm(geometry,hm):    
    for polygons in hm['.geo']:
        if(inside_geom(geometry,polygons)):
            return polygons
        

with open('Data\\KNM.txt','r') as f:
    r = f.readlines() 

mf = 0
for line in r:
    geom  = line.split("#")[0].split('_')
    
    json.loads(line[:-1].split('#')[1]) | 
    
    #check_respective_hm(geom, hm)
    if(mf%10 ==0):
        print(mf/len(r))
    mf += 1
    
    get_respective_hm(geom,hm)
    
    