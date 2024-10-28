# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:38:10 2024

@author: kashy
"""

'''
This file has the entire pipeline. 
Combines all the functionalities in all files. 


#Never do such things, as they might have unintended effects
#When you change the config file
if 'saved' in dictionary['DEFAULTS']['run']:
    pass

#Always, do this
if dictionary['DEFAULTS']['run'] in ['saved','saved_redo']:
    try:
        with open('blahblah.csv') as f:
            pass
    except:
        if(dictionary['DEFAULTS']['run'] == 'saved'):
            raise FileNotFoundError("run is set to saved in config and file was not found.")
'''
            
import yaml
import numpy as np
import pandas as pd
import math 
import re
import warnings 


####

from preprocessing.Cloudfill import cloudmask


import GlobalVars
from utils import utils 

    
def execute_pipeline_item(item ,config, saved_file = None,input_file=None,params = None,save_path=None):
    '''
    This execute any item of the pipeline. 
    Essential arguments are : 
        1. name of the item. this is a tuple of (name,package). Name is the same on in config['FUNCTIONS']
        2. Config dictionary 
            2.1 runtype  - Config['DEFAULTS']['run']
            2.2 after_run - Config['DEFAULTS']['after_run']
    Optional arguments are: 
        1. saved file - This is to be read if we don't have to execute the item. 
        2. input file - This is to be used if we are executing the item 
        Either optional have to be sent. 
        3. Params. It is better to be sent as argument as they can be overridden based on requirement.
        4. save path to save. 
        
    -> After run, decide to save based on config file 
    -> When adding functionality to pass, make sure to not make errors. 
    '''   
    redo = False
    if(config['DEFAULTS']['run'] in ['saved' , 'saved_redo'] ):
        
        success , retval =  utils.safe_load(saved_file)
        if(not success):
            if(config['DEFAULTS']['run'] == 'saved'):
                raise Exception(f"Run option is <<saved>> but Cloud file wasn't found at {GlobalVars.Cloud_file} ")
            elif(config['DEFAULTS']['run'] == 'saved_redo'):
                redo = True
            else:
                raise Exception(f"Unknown option for Config-DEFAULTS-run: {config['DEFAULTS']['run']}")
        else:
            return retval
        
    elif(redo or config['DEFAULTS']['run']=='redo'):
        redo = True # this is important for saving. 
        itemname , pack = item 
        if(not isinstance(input_file, (pd.DataFrame, list, dict))) :
            raise Exception(f"Trying to execute function <<{itemname}>> but input file is dataframe.")
        params['first_arg'] = input_file
        retval = pack.pipeline_executable(**params)
        return retval
        
    elif(config['DEFAULTS']['run'] in ['redo_saved','manual']):
        raise NotImplementedError(f"The Runtype, <<{config['DEFAULTS']['run']}>> hasn't been implemented yet.")
    else:
        raise Exception(f"Runtype <<{config['DEFAULTS']['run']}>> isn't recognized.")
        
    if(config['DEFAULTS']['after_run'] == 'save'):
        if(redo):
            print("Not saving file that was only read.")
        retval.to_csv(save_path)
        

def execute(config):
    if(config['GDD']):
        hm,optical ,centroid_temperatures = prerequisites(config,gdd=True)
    else:
        hm,optical = prerequisites(config,gdd=False)
        
        # You can avoid null value problems by dropping SowDate for non GDD
        optical = optical.drop(columns= ['Sow_Date'])
        
    # 1. Cloud Masking/Filling
    import preprocessing.Cloudfill as cl
    item = ('CloudCorrection',cl)
    params = config['FUNCTIONS']['CloudCorrection']['params']
    
    cloudmasked = execute_pipeline_item(item , config , 
                                        saved_file = GlobalVars.Cloud_file,
                                        input_file = optical,
                                        params = params,
                                        save_path = 'Data/Interim/Cloud/Optical_Cloudfilled_'+dictionary['GLOBALNAME']+'.csv')
    
    
    
    
    # 2.1 CGDD 
    if(config['GDD']):
        raise Exception("This has been implemented but didn't add into the main pipeline execution.")
    
    # 2.2 Harmonised Time Composite
    if(not config['GDD']):
        import Harmonised_Time_Composite as htm 
        item  = ('HarmonisedTimeComposite',htm)
        params = config['FUNCTIONS']['HarmonisedTimeComposite']['params'] 
        params['bands'] = GlobalVars.optical_bands 
        fixed  , increment = params['fixed'],params['increment']
        which = 'Optical'
        if(fixed):
            save_path = 'Data\\interim\\Satellite\\all_'+which+f'_{increment}day_fixedJan1_'+dictionary['GLOBALNAME']+'.csv'
        else:
            save_path = 'Data\\interim\\Satellite\\all_'+which+f'_{increment}day_'+dictionary['GLOBALNAME']+'.csv'
        
        composited = execute_pipeline_item(item, config,
                                           input_file = cloudmasked,
                                           params = params , 
                                           save_path = save_path) 
    
    # 3. PreML
    ## 3.1 Preprocessing 
    import preprocessing.Preprocess as pp
    item = ('Preprocess',pp)
    params = config['FUNCTIONS']['Preprocess']['params'] 
    preprocessed = execute_pipeline_item(item,config,
                                         input_file = composited ,
                                         params = params ,
                                         save_path ='Data/Interim/Preprocessed/Optical_Cloudfilled_preprocessed_'+dictionary['GLOBALNAME']+'.csv')

    ## 3.2 Feature Addition 
    
    import preprocessing.FeatureAddition as pfa 
    item = ('FeatureAddition',pfa)
    params = config['FUNCTIONS']['FeatureAddition']['params'] 
    config['FUNCTIONS']['FeatureAddition']['params']['tim'] = True
    
    feature_added = execute_pipeline_item(item,config,
                                         input_file = composited ,
                                         params = params ,
                                         save_path ='Data/Interim/Added/Optical_Cloudfilled_preprocessed_added_'+dictionary['GLOBALNAME']+'.csv')
    

    
    
    ### 3.3 Feature selection 
    
    
    
    
    ### 4 ML
    
    
    
        
        
        

    
def get_base_config_dictionary(fname):
    with open(fname) as stream:
        try:
            dictionary = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return dictionary 

def ConfigWrapper(fname):
    '''
    Load config, overwrite changes with special arguments. 
    '''
    dictionary  = get_base_config_dictionary(fname)
    # no overwriting changes currently.
    # But, I am not implementing until GeneratingCentroidTemperatures. 
    dictionary['GLOBALNAME'] = 'Beta001'
    
    warnings.warn("CHANGING CONFIG",UserWarning)
    return dictionary 



def prerequisites(config , gdd=True):
    # List of data that's essential for the pipeline.
    hm =  pd.read_csv(GlobalVars.harmonised_file)
    optical = pd.read_csv(GlobalVars.optical_file)
    
    if(gdd):
        centroid_temperatures = pd.read_csv(GlobalVars.CentroidTemperatures_file)
        return hm,optical, centroid_temperatures
    return hm,optical
    
    
    


    
    
    
    
if __name__ == '__main__':
    dictionary = ConfigWrapper('config/config.yaml')
    execute(dictionary)
    
    