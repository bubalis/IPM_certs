#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:42:02 2021

@author: bdube
"""
import geopandas as gpd
from multiprocessing import  Pool
from functools import partial
import pandas as pd
import rasterio as rio
import os
import numpy as np
from rasterstats import zonal_stats
import re

m2_acre_conv = .0002477105


def max_area_attr_join(gdf1, gdf2, id_col, attribute_col):
    '''Add attribute {attribute_col} from gdf2 to gdf1, based on
    the largest area.
    '''
    
    s = gpd.overlay(gdf1, gdf2, how = 'intersection') 
    s['area_ov'] = s.geometry.area
    
    gb = s.groupby(id_col)[['area_ov']].max()
    gdf1 = gdf1.merge(gb, left_on = id_col, right_index = True) 
    
    s = s[['area_ov', attribute_col, id_col]]
    gdf1 = gdf1.merge(s, on = ['area_ov', id_col])
    return gdf1.drop(columns = ['area_ov'])


def get_cellsize(rast):
    '''Get the area of a raster cell.
    Args: a dataset reader raster.'''
    xform = rast.get_transform()
    return xform[1]*xform[-1]*-1

def dissolve_all(gdf):
    gdf['null'] = 0
    return gdf.dissolve('null')


def total_area_eq_val(rast_fp, gdf, val = 69):
    '''Return the total area (in acres) of cells where the raster value = val.
    for a given raster and gdf. '''
    
    cell_size = get_cellsize(rio.open(rast_fp))
    print(cell_size)
    res = zonal_stats(gdf, rast_fp, categorical = True)
    
    out_data = [r.get(val, 0)*cell_size*m2_acre_conv for r in res]
    return out_data


def get_rast_crs(rast_fp):
    return rio.open(rast_fp).crs

def get_all_areas(source_dir, gdf, 
                  col_name_str= 'acres_grapes_', val = 69,
                  rast_filetype = '.tif'):
    '''Get areas (in acres) within shapes where raster value = val. 
    Add those values as columns in the gdf. Do this for all rasters in 
    source_dir return the gdf. 
    '''
    orig_crs = gdf.crs 
    raster_files = [f for f in os.listdir(source_dir) if re.search(f'{rast_filetype}$', f)]
    for i, file in enumerate(raster_files):
        fp = os.path.join(source_dir, file)
        if i == 0:
            rast_crs = get_rast_crs(fp)
            if orig_crs != rast_crs:
                gdf.to_crs(rast_crs, inplace = True)
            
        col_name = col_name_str+ file.split('_')[1].replace(rast_filetype, '')
        gdf[col_name] = total_area_eq_val(fp, gdf)
    
    return gdf.to_crs(orig_crs)



def intersects(gdf1, gdf2):
    '''Return all elements of gdf that touch an element of gdf2'''
    
    gdf1['scratch_id'] = list(range(gdf1.shape[0]))
    gdf3 = gpd.overlay(gdf1, gdf2)
    gdf1 = gdf1[gdf1['scratch_id'].isin(gdf3['scratch_id'].unique())]
    return gdf1.drop(columns = ['scratch_id'])




def parallelize(data, func, num_of_processes=8):
    '''Function for paralellizing any function on a dataframe.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply'''
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data



def run_on_subset(func, data_subset):
    '''For use in parallelize.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply'''
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    '''Apply a function to every row in a pandas df in parallell.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply''' 
    return parallelize(data, partial(run_on_subset, func), num_of_processes)