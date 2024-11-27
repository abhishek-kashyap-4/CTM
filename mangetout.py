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

    
def execute_pipeline_item(item ,config, saved_file = None,input_file=None,params = None,save_path=''):
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
    
    
    if(config['FUNCTIONS'][item[0]]['skip']):
        return  []

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
            pass
        
    if(redo or config['DEFAULTS']['run']=='redo'):
        #A_K_ actually this may not be correct.
        redo = True # this is important for saving. 
        itemname , pack = item 
        if(not isinstance(input_file, (pd.DataFrame, list, dict))) :
            raise Exception(f"Trying to execute function <<{itemname}>> but input file is dataframe.")
        params['first_arg'] = input_file
        retval = pack.pipeline_executable(**params)
        
    elif(config['DEFAULTS']['run'] in ['redo_saved','manual']):
        raise NotImplementedError(f"The Runtype, <<{config['DEFAULTS']['run']}>> hasn't been implemented yet.")
    else:
        # This is important , because its not a full if else chain.
        if(config['DEFAULTS']['run'] not in ['saved_redo']):
            raise Exception(f"Runtype <<{config['DEFAULTS']['run']}>> isn't recognized.")
        
    if(config['DEFAULTS']['after_run'] == 'save'):
        if(not redo):
            print("Not saving file that was only read.")
        if(len(save_path)==0):
            print("Save path not provided. Skipping save.")
        else:
            retval.to_csv(save_path)
    return retval
        

def execute_mangetout(config):
    '''
    This is different from execute in main. 
    -> use config['additional'] to do additional stuff - like sampling the df, 
             - Take the title to use for plots (this is usecase - like summer vs winter)
    '''
    
    if(config['GDD']):
        hm,optical ,hm_temperatures = prerequisites(config,gdd=True)
    else:
        hm,optical = prerequisites(config,gdd=False)
        
        # You can avoid null value problems by dropping SowDate for non GDD
        #optical = optical.drop(columns= ['Sow_Date'])
    optical = optical.sample(n = config['additional']['n']) 
    optical.reset_index(inplace=True,drop=True)
    title = config['additional']['title']
    datakind = config['datakind']
    print(f"Executing for Usecase: {title} , with datakind {datakind} and GDD {config['GDD']}")
    
    
    optical = utils.fix_column_syntax(optical,from_re = '[0-9]{8}_')
    
    
    # 1. Cloud Masking/Filling
    import preprocessing.Cloudfill as cl
    item = ('CloudCorrection',cl)
    params = config['FUNCTIONS']['CloudCorrection']['params']
    cloudmasked = execute_pipeline_item(item , config , 
                                        input_file = optical.copy(),
                                        params = params,
                                        save_path = 'Data/Interim/Cloud/Optical_Cloudfilled_'+dictionary['GLOBALNAME_OUTPUT']+'.csv')
    
    
    if(len(cloudmasked) > 0):
        utils.check_column_syntax(cloudmasked ,kind='date')
    
    # 2.1 CGDD 
    if(config['GDD']):
        if(config['SUPPORT']['GDD_Version'] == 'new'):
            import compositing.GDDComposite_new as gddn 
            item = ('GDDComposite_new',gddn)
            params = config['FUNCTIONS']['GDDComposite_new']['params']
            params['hm_temperatures'] = hm_temperatures
            params['hm'] = hm
            save_path = 'Data/Interim/CGDD/Optical_'+dictionary['GLOBALNAME_OUTPUT']+'.csv' 
            composited = execute_pipeline_item(item,config,
                                               saved_file = save_path,
                                               input_file = optical.copy() , 
                                               params = params , 
                                               save_path = save_path ) 
            
        
        elif(config['SUPPORT']['GDD_Version'] == 'old'):
            import compositing.GenerateCentroidTemperaturesCGDD as gcc 
            item  = ('GenerateCentroidTemperaturesCGDD' , gcc)
            params =  config['FUNCTIONS']['GenerateCentroidTemperaturesCGDD']['params']
            centroids  = pd.read_csv('Data/Interim/post/temperature_V5_2023_centroids.csv')
            if(params['method'] == 'fixed'):
                save_path = 'Data/Interim/Post/centroid_temperatures_cgdd_v5_fixed_'+str(params['startdate'])+'_'+dictionary['GLOBALNAME_OUTPUT']+'.csv' 
            else:
                save_path = 'Data/Interim/Post/centroid_temperatures_cgdd_v5_dynamic_'+dictionary['GLOBALNAME_OUTPUT']+'.csv' 
            cgdd = execute_pipeline_item(item , config , 
                                                input_file = centroids.copy(),
                                                params = params,
                                                save_path = save_path)
                        #####               #######             ######
            
            
            import compositing.HarmonisedGDDComposite as HGC 
            item = ('HarmonisedGDDComposite',HGC)
            params = config['FUNCTIONS']['HarmonisedGDDComposite']['params']
            params['hm'] = hm.copy()
            params['cgdd'] = cgdd.copy()
            params['bands'] = GlobalVars.optical_bands
            save_path = 'Data/Interim/CGDD/cgdd_'+dictionary['GLOBALNAME_OUTPUT']+'.csv'
            composited = execute_pipeline_item(item , config , 
                                                input_file = optical.copy(),
                                                params = params,
                                                save_path = save_path)
            
            
        
        utils.check_column_syntax(composited , kind = 'timestep',stricter = True)
        
        
    # 2.2 Harmonised Time Composite
    elif(not config['GDD']):
        import compositing.Harmonised_Time_Composite as htm 
        item  = ('HarmonisedTimeComposite',htm)
        params = config['FUNCTIONS']['HarmonisedTimeComposite']['params'] 
        params['bands'] = GlobalVars.optical_bands 
        fixed  , increment = params['fixed'],params['increment']
        which = 'Optical'
        if(fixed):
            save_path = 'Data\\Interim\\Satellite\\all_'+which+f'_{increment}day_fixedJan1_'+dictionary['GLOBALNAME_OUTPUT']+'.csv'
        else:
            save_path = 'Data\\Interim\\Satellite\\all_'+which+f'_{increment}day_'+dictionary['GLOBALNAME_OUTPUT']+'.csv'
        
        composited = execute_pipeline_item(item, config,
                                           input_file = optical.copy(),
                                           params = params , 
                                          save_path = save_path) 
        utils.check_column_syntax(composited , kind = 'timestep',stricter = True)
        
        
    # Compositing postprocess.   
    if(config['datakind'] in ['annotated' ]):
        composited = utils.add_hm_features(composited,hm,features = ['Crop_Type','lat'])
    else:
        composited = utils.add_hm_features(composited,hm,features = ['Crop_Type','Sow_Date'])
    
    #composited = composited[composited.Sow_Date.notna()]
    utils.check_column_syntax(composited ,kind='timestep')
    
    
    # 3. PreML
    ## 3.1 Preprocessing 
    import preprocessing.Preprocess as pp 
    item = ('Preprocess',pp)
    params = config['FUNCTIONS']['Preprocess']['params'] 
    
    if(config['datakind'] in ['annotated']):
        params['mapper'] = GlobalVars.target_remap_annotated
        composited['Sow_Date'] = np.nan
        
        
    preprocessed = execute_pipeline_item(item,config,
                                         input_file = composited.copy() ,
                                         params = params ,
                                         save_path ='Data/Interim/Preprocessed/Optical_'+dictionary['GLOBALNAME_OUTPUT']+'.csv')

    ## 3.2 Feature Addition 
    
    import preprocessing.FeatureAddition as pfa 
    item = ('FeatureAddition',pfa)
    params = config['FUNCTIONS']['FeatureAddition']['params'] 
    #config['FUNCTIONS']['FeatureAddition']['params']['tim'] = True
    
    feature_added = execute_pipeline_item(item,config,
                                         input_file = preprocessed.copy(),
                                         params = params ,
                                         save_path ='Data/Interim/Added/Optical_'+dictionary['GLOBALNAME_OUTPUT']+'.csv')
    

    feature_added.to_csv('dudcking.csv')
    
    if(config['GDD']):
        pass 
    
    import EDA.EDA_Functions as eda 
    eda.band_series_by_croptype(feature_added,'NDVI',method='median')
    for croptype in np.unique(feature_added.Crop_Type):
        eda.plot_mean_std(feature_added , bands = ['NDVI'],croptypes = [croptype])

    eda.plot_all_signals(feature_added , band = 'NDVI')
    
    
    print('tmax var',feature_added.t_max_NDVI.var())
    
  
def execute_tuner(config):
    '''
    -> Code to get all related graphs. 
    -> Execute Seperately for both data versions as globalvars has conditional filenames.
            - Also, configWrapper(executed before) is already checking for datakind.    
    -> Tune here. Call Execute() to run everything.
    '''
    config['FUNCTIONS']['FeatureAddition']['params']['tim'] = False
    config['DEFAULTS']['run'] = 'redo'
    config['DEFAULTS']['after_run'] = 'pass'
    
    
    titles = ['normal']
    datakinds = ['veredi','annotated']
    gdds = [True,False]
    
    config['additional'] = {}
    config['additional']['n'] = 200
    
    for title in titles:
        config['additional']['title'] = title 
        for datakind in datakinds:
            config['datakind'] = datakind 
            for gdd in gdds:
                config['GDD'] = gdd 
                execute_mangetout(config)
                3/0
    #execute_mangetout(config)
    





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
    assert dictionary['datakind'] == GlobalVars.which, "Data version incompatibility"
    # no overwriting changes currently.
    # But, I am not implementing until GeneratingCentroidTemperatures. 
    
    
    #warnings.warn("CHANGING CONFIG",UserWarning)
    return dictionary 



def prerequisites(config , gdd=True):
    # List of data that's essential for the pipeline.
    hm =  pd.read_csv(GlobalVars.harmonised_file)
    optical = pd.read_csv(GlobalVars.optical_file)
    if(config['datakind'] in ['annotated']):
        hm = hm.rename(columns={'Class_st':'Crop_Type'})
    if(gdd):
        hm_temperatures = pd.read_csv(GlobalVars.hm_temperatures_file)
        return hm,optical, hm_temperatures
    return hm,optical
    

if __name__ == '__main__':
    
    dictionary = ConfigWrapper('config/config.yaml')
    
    execute_tuner(dictionary)
    
    