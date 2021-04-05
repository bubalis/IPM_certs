#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:33:34 2021

@author: bdube
"""
import geopandas as gpd
import pandas as pd
import os
import json
import rasterio as rio
import json
import multiprocessing as mp
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
#from rasterio.enums import Resampling
from spatial_utils import parallelize_on_rows
import numpy as np
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
from math import inf
from functools  import partial


#%%
eiq_parts_cols =['Farm.Worker', 'Consumer', 'Ecology', 'EIQ.Value']

action_dict = dict(
    AC = 'Insecticide',
    I = 'Insecticide',
    IGR = 'Insecticide',
    BP = 'Insecticide',
    BF = 'Insecticide',
    Bac = 'Insecticide',
    F = 'Fungicide',
    H = 'Herbicide',
    Fum = 'Fumigant',
    PGR = 'Other',
    PA = 'Other',
    B = 'Other',
    CP = 'Other',
    R = 'Rodenticide',
    
    )






#%%



#%%
fumigants = ['methyl bromide', 'metam potassium', 'metam sodium',
             'dazomet', '1,3-dichloropropene', 'chloropicrin']

def check_rec_bad(row):
    '''Guidelines here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.168.6558&rep=rep1&type=pdf
    for determining if records are bad.'''
    try:
        return any([(row['LBS_PER_AC'] > row['median_rate']*50),
            ((row['CHEMNAME'] not in fumigants) & (row['LBS_PER_AC']>200)),
            ((row['CHEMNAME'] in fumigants) & (row['LBS_PER_AC']>1000) )])
    except:
        print(row)
        raise
    
def fix_bad_recs(apps):
    '''Apply the guideline here: that applications that are 50x the median
    must be in error. '''
    
    medians = apps.groupby('CHEMNAME')['LBS_PER_AC'].median().to_dict()
    #with mp.Pool(4) as pool:
    #    apps['median_rate'] = pool.map(medians.get, apps['CHEMNAME'])
    apps['median_rate'] = apps['CHEMNAME'].apply(medians.get )
    indicies = apps.apply(check_rec_bad, axis = 1)
    #indicies = parallelize_on_rows(apps, check_rec_bad, 4)
    
    apps.loc[indicies, "LBS_AI"] = apps.loc[indicies, 'median_rate'] * apps.loc[indicies, 'AREA_TREATED']
    apps['LBS_PER_AC'] = apps['LBS_AI']/apps['AREA_TREATED']
    return apps, indicies



def assign_lookup_name(name, lookup_names):
    if type(name)!= str:
        return None
    if name in lookup_names:
        return name
    elif name.split()[0] in lookup_names:
        return name.split()[0]
    elif any([name in n for n in lookup_names]):
        return [n for n in lookup_names if name in n][0]
    elif any([n in name for n in lookup_names]):
        return [n for n in lookup_names if n in name][0]
    elif any(['bacillus' in name,
              ' strain ' in name,]
              ):
        return 'bacteria_other'
    else:
        return 'not_included'

def clean_chemname(string):
    string = str(string).lower()
    return string.replace('-', '')

def dic_get_same(dic, key):
    return dic.get(key, key)

def load_apps():
    apps = pd.read_csv(os.path.join('source_data', 'PIP_recs.txt'), sep ='\t')
    apps['PRODUCT_NAME']= apps['PRODUCT_NAME'].str.lower()
    eiq = pd.read_csv(os.path.join('source_data',  'eiq_values.csv'), encoding = 'latin-1')
    
    
    #with mp.Pool(4) as pool:
        #apps['CHEMNAME'] =  pool.map(clean_chemname, apps['CHEMNAME'])
        #eiq['Common.Name'] = pool.map(clean_chemname, eiq['Common.Name'])
        #lookup_names = eiq['Common.Name'].unique()
        #lookup_func = partial(assign_lookup_name, lookup_names=lookup_names)
       
        
        #lookup2 = partial(dic_get_same, action_dict)
        #eiq['Type'] = pool.map(lookup2, eiq['Action'])
    
    apps ['CHEMNAME'] = apps['CHEMNAME'].apply(clean_chemname)
    eiq['Common.Name'] = eiq['Common.Name'].apply(clean_chemname)
    lookup_names = eiq['Common.Name'].unique()
    lookup_func = partial(assign_lookup_name, lookup_names=lookup_names)
    apps['LOOKUP_NAME'] =  apps['CHEMNAME'].apply(lookup_func) 
    lookup2 = partial(dic_get_same, action_dict)
    eiq['Type'] =  eiq['Action'].apply(lookup2,)
    
    apps['LBS_PER_AC'] = apps['LBS_AI']/apps['AREA_TREATED']
    
    apps, fixed_indicies = fix_bad_recs(apps)
    
    
    
    
    
   
    
    
    apps = apps.merge(eiq[['Common.Name', 'Action', 'Type'] +eiq_parts_cols],
                      left_on = 'LOOKUP_NAME', right_on= 'Common.Name', how= 'left')

    for column in eiq_parts_cols:
        apps[column] = apps[column]*apps['LBS_AI']
    
    return apps, eiq

if __name__ == '__main__':
    os.chdir('spatial')
    apps, eiq = load_apps()

    lodi_comtrs = gpd.read_file(os.path.join('intermed_data', 'lodi_comtrs'))
    
    grape_data = pd.read_csv(os.path.join('intermed_data', 'lodi_grape_acres.csv')).drop(columns = ['Unnamed: 0'])
    grape_tidy = grape_data.melt(id_vars = ['COMTRS'])
    grape_tidy.rename(columns = {'variable': 'Year', 'value': 'acres_grapes'}, inplace = True)
    grape_tidy['Year'] = grape_tidy['Year'].apply(lambda x: int(x.split('_')[-1]))
          
    totals = apps.groupby(['YEAR','COMTRS'])[['LBS_AI']+eiq_parts_cols].sum().reset_index()
    
    grape_tidy = grape_tidy.merge(totals, left_on= ['Year', 'COMTRS'], 
                                  right_on = ['YEAR', 'COMTRS'], how= 'left')
    grape_tidy.drop(columns = 'YEAR', inplace =True)
    
    
    years = grape_tidy.groupby('Year')[['acres_grapes', 'LBS_AI']+eiq_parts_cols].sum()
    
    years['LBS_AI'].plot()
    plt.show()
    
    grape_tidy.groupby('Year')['acres_grapes'].sum().plot()
    plt.show()
    years['acres_grapes'] = grape_tidy.groupby('Year')['acres_grapes'].sum()
    years['lbs_per_acre'] = (years['LBS_AI']/years['acres_grapes']).replace({inf: 0})
    years['lbs_per_acre'].loc[1990:].plot()
    #%%
    
    
    comtrs = gpd.read_file(os.path.join('source_data', 'comtrs'))
    
    '''
    data_cols = sorted([c for c in grape_data.columns if 'acres_grapes' in c], key = lambda x: int(x.split('_')[-1]))
    
    grape_data_smooth = grape_data
    grape_data_smooth[data_cols] = grape_data_smooth[data_cols].T.rolling(4).mean().T
    for col in data_cols:
        grape_data_smooth[col] = grape_data_smooth[col].apply(lambda x: x if x>=1 else 0)         
                                                                
    grape_tidy_smth = grape_data.melt(id_vars = ['COMTRS'])
    grape_tidy_smth.rename(columns = {'variable': 'Year', 'value': 'acres_grapes'}, inplace = True)
    grape_tidy_smth['Year'] = grape_tidy_smth['Year'].apply(lambda x: int(x.split('_')[-1]))
    
    
    
    grape_tidy_smth = grape_tidy_smth.merge(totals.dropna(subset = ['COMTRS']), left_on= ['Year', 'COMTRS'], 
                                  right_on = ['YEAR', 'COMTRS'], how= 'left')
    grape_tidy_smth.drop(columns = 'YEAR', inplace =True)
    
    grape_tidy_smth['lbs_per_acre'] = grape_tidy_smth['LBS_AI']/grape_tidy_smth['acres_grapes']
    grape_tidy_smth['lbs_per_acre'] = grape_tidy_smth['lbs_per_acre'].replace({inf: 0})
    
    smth_years = grape_tidy_smth.groupby('Year').sum()
    smth_years['acres_grapes'].plot()
    smth_years['lbs_per_acre'] = (smth_years['LBS_AI']/smth_years['acres_grapes']).replace({inf: 0})



for i in range(2000, 2020):
    subset = grape_tidy_smth[grape_tidy_smth['Year']==i]
    print(np.quantile(subset['lbs_per_acre'].dropna(), .99)) 
    print(subset['lbs_per_acre'].mean())
    print('\n')
    
    
    
apps['Type'] = apps['Action'].apply(lambda x: action_dict.get(x, x))

def norm(a):
    return (a - a.mean()) / a.std()


gb = apps.groupby(['Type', 'YEAR'])[eiq_parts_cols+['LBS_AI']].sum()


for i in gb.index.levels[0]:
    norm(gb.loc[i, 'LBS_AI']).plot()
    plt.legend(list(gb.index.levels[0]))

#%%
for i in gb.index.levels[0]:
    norm(gb.loc[i, 'Consumer']).plot()
    norm(gb.loc[i, 'Ecology']).plot()
    norm(gb.loc[i, 'Farm.Worker']).plot()
    norm(gb.loc[i, 'LBS_AI']).plot()
    plt.legend(['Consumer', 'Ecology', 'FarmWorker', "TOTAL AI"])
    plt.title(i)
    plt.show()
    
gb2 = apps.groupby(['Type', 'COMTRS', 'YEAR', ])[eiq_parts_cols+['LBS_AI']].sum()
#%%
for i in gb2.index.levels[0]:
    for val in eiq_parts_cols:
        sub = gb2.loc[i]
        x = sub['LBS_AI']
        y = sub[val]
        x = x[:, np.newaxis]
        plt.scatter(x,y, alpha =.4)
        a, _ , _, _ = np.linalg.lstsq(x, y)
        pred_x = np.linspace(0, max(x), 500)
        plt.plot(pred_x, pred_x*a,  color = 'k')
        plt.title(f'LBS_AI vs {val} for {i}' )
        plt.show()
#%%
for i in gb.index.levels[0]:
    sub = gb2.loc[i]
    sns.pairplot(sub)
    plt.title(f'Correlations for {i}')
'''    