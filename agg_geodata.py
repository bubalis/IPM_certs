#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:32:39 2021

@author: bdube
"""
import geopandas as gpd
import os
import re
#from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

from spatial_utils import m2_acre_conv, dissolve_all, get_all_areas




os.chdir('spatial')
def interpolate_acres_grapes(grape_data):
    pass

def multi_select(gdf, dic):
    return gdf[gdf.apply(lambda row: np.any([row[key]==value for key, value in dic.items()]), axis = 1)]

def overlay_get_area(gdf, fp, match_dict, agg_column, new_col_name):
    
    data = gpd.read_file(fp).to_crs(gdf.crs)
    bounds = dissolve_all(data[data['CLASS1']!='Z']).geometry
    ov = gpd.overlay(data, gdf)
    ov = multi_select(ov, match_dict)
    ov['area_new'] = ov.geometry.area
    areas = ov.groupby(agg_column)[['area_new']].sum() * m2_acre_conv
    gdf = gdf.merge(areas, left_on = agg_column, right_index = True, how = 'left',
                    )
    gdf.loc[(gdf['area_new'].isna()) & (gdf.geometry.intersects(bounds.iloc[0])),
                'area_new'] = 0
    return gdf.rename(columns= {'area_new': new_col_name})





#%%
def interpolate_acres_grapes(grape_data):
    pass

def multi_select(gdf, dic):
    return gdf[gdf.apply(lambda row: np.any([row[key]==value for key, value in dic.items()]), axis = 1)]

def overlay_get_area(gdf, fp, match_dict, agg_column, new_col_name):
    
    data = gpd.read_file(fp).to_crs(gdf.crs)
    bounds = dissolve_all(data[data['CLASS1']!='Z']).geometry
    ov = gpd.overlay(data, gdf)
    ov = multi_select(ov, match_dict)
    ov['area_new'] = ov.geometry.area
    areas = ov.groupby(agg_column)[['area_new']].sum() * m2_acre_conv
    gdf = gdf.merge(areas, left_on = agg_column, right_index = True, how = 'left',
                    )
    gdf.loc[(gdf['area_new'].isna()) & (gdf.geometry.intersects(bounds.iloc[0])),
                'area_new'] = 0
    return gdf.rename(columns= {'area_new': new_col_name})

import re 

def finish_year(string):
    if int(string)< 30:
        return '20'+ string
    else:
        return '19' + string





if __name__ == '__main__':
    lodi_comtrs = gpd.read_file(os.path.join('intermed_data', 'lodi_comtrs'))
    
    source_dir = os.path.join('source_data', 'Lodi_CDL')
    
        
    lodi_comtrs = lodi_comtrs.dissolve(by= 'NAME').reset_index()
    lodi_comtrs= lodi_comtrs[lodi_comtrs['NAME'].isin(['San Joaquin', 'Sacramento'])]
    lodi_comtrs = get_all_areas(source_dir, lodi_comtrs)
    
    
        
        
    
    
    #%%
  
    #grape_data.set_index('COMTRS', inplace = True)
    
    #linear interpolation of missing acreage data:
    #grape_data[data_cols] = grape_data[data_cols].T.reset_index().interpolate('linear').T
    
    
       
    

    
    #%%
    
    
    for fn in [f for f in os.listdir('source_data') if re.search('^\d\d\w\w$',f)]:
        n = fn[:2]
        fp = os.path.join('source_data', fn)
        lodi_comtrs = overlay_get_area(lodi_comtrs, 
                                   fp = fp,
                                   match_dict = dict(CLASS1 = 'V', CLASS2 = 'V'
                                       ),
                                   agg_column = 'NAME',
                                   new_col_name =  f'acres_grapes_{finish_year(n)}')
        
    
    for i in range(1988, 2020):
        col_name = f'acres_grapes_{i}'
        if col_name not in lodi_comtrs.columns:
            lodi_comtrs[col_name] = np.nan
            
    
    #%%
    
    data_cols = sorted([c for c in lodi_comtrs.columns if 'acres_grapes' in c], key = lambda x: int(x.split('_')[-1]))
    for c in data_cols:
        lodi_comtrs[c] =  lodi_comtrs[c].apply(lambda x: x if x>1000 else np.nan).astype(float)
    grape_data = lodi_comtrs[['NAME']+data_cols]
    grape_data[data_cols] = grape_data[data_cols].T.reset_index().interpolate('linear').T
    grape_data = grape_data.melt(id_vars = ['NAME']).rename(columns = {'variable': 'Year', 'value': 'acres_grapes'})
    grape_data['Year'] = grape_data['Year'].apply(lambda x: int(x.split('_')[-1]))
    
    #grape_data.set_index('COMTRS', inplace = True)
    #linear interpolation of missing acreage data:
    grape_data = grape_data[['acres_grapes', 'Year']].groupby('Year').sum().reset_index()
        
    #grape_data['acres_grapes'] = grape_data['acres_grapes'].interpolate('linear')
    grape_data['acres_grapes'] = grape_data['acres_grapes'].rolling(5).mean()
    grape_data.to_csv(os.path.join('intermed_data', 'lodi_total_grape_acres.csv'))
    
    #del grape_data
    #del lodi_comtrs
   
    
    #%%   
    comtrs = gpd.read_file(os.path.join('intermed_data', 'ava_comtrs'))
    

    
   
    source_dir = os.path.join('source_data', 'Cali_CDL')
    
    comtrs = get_all_areas(source_dir, comtrs)
    data_cols = sorted([c for c in comtrs.columns if 'acres_grapes' in c], key = lambda x: int(x.split('_')[-1]))
    grape_data = comtrs[['COMTRS']+data_cols]
    grape_data.to_csv(os.path.join('intermed_data', 'california_grape_acres.csv'))
    