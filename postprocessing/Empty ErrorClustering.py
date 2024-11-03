# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:34:06 2024

@author: kashy
"""

'''
Error clustering is an algorithm which analysis the error rows and how they are linked with each other. 

It tries to cluster similar errors together. 
Decision needs to be taken about whether it is a training error or a test error. 

In a single run, similar rows needn't be grouped together as they might have been a part of training
(Provided you are keeping train and test seperate for this.)

Basically, a row is similar to another if more models error out on both at the same time.

Another decision to be taken would be - 
    Do you consider a point, or a row as a single entity? If feature engineering was done differently, 
    Is it a different point? 
    
Remember that you should only consider 'good' models, not all of them.
'''
