#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:25:12 2021

@author: bdube
"""
import geopandas as gpd
import os
import rasterio as rio
import json
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
#from rasterio.enums import Resampling
from spatial_utils import  dissolve_all, get_rast_crs, intersects, max_area_attr_join
import numpy as np
import pandas as pd



'''
def make_grape_rasters(source_dir, dst_dir, dst_crs, start_year=2007, stop_year = 2020
                       ):
    for year in range(start_year, stop_year):
        src_fp = [f for f in os.listdir(source_dir) if f'CDL_{year}' in f and f[-4:]=='.tif'][0] 
        src_fp = os.path.join(source_dir, src_fp)
        dst_fp1 = os.path.join(dst_dir, f'CDL_{year}_reproj.tif')
        dst_fp2 = os.path.join(dst_dir, f'CDL_{year}_is_grape.tif')
        
        with rio.open(src_fp) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
        
            with rio.open(dst_fp1, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
                    
        with rio.open(dst_fp1, 'r', ) as src:
            with rio.open(dst_fp2, 'w+', **kwargs) as dst:
                dst.write(np.array([np.where(src.read(1) == 69, 1, 0  )]).astype(kwargs['dtype']))
                
'''
   #%%
   
if __name__ == '__main__':
    #os.chdir('spatial')
    
    lodi_data = json.loads(open(os.path.join('source_data', 'lodi_data.txt')).read())
    
    counties = gpd.read_file(os.path.join('source_data', 'ca-county-boundaries', 'CA_Counties'))
    lodi_bounds = gpd.read_file(os.path.join('source_data', 'lodi_boundary'))
    lodi_bounds.to_crs(counties.crs, inplace = True)
    lodi_counties = gpd.overlay(counties, lodi_bounds)
    
    vineyards = gpd.GeoDataFrame(data = [key for key in lodi_data],                
                                 geometry= gpd.points_from_xy([float(entry['lon']) for entry in lodi_data.values()],
                             [float(entry['lat']) for entry in lodi_data.values()]), 
                                crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    vineyards.rename(columns= {0: 'Name'}, inplace=True)
    vineyards.to_crs(lodi_bounds.crs,inplace=True)
    
    lodi_vineyards = gpd.clip(vineyards, lodi_bounds)
    
    
    comtrs = gpd.read_file(os.path.join('source_data', 'comtrs'))
    comtrs.to_crs(lodi_bounds.crs)
    lodi_comtrs = comtrs[comtrs.geometry.intersects(lodi_bounds.geometry.iloc[0])]
    #lodi_comtrs = lodi_comtrs[lodi_comtrs.geometry.area>10000]
    lodi_comtrs.to_file(os.path.join('intermed_data', 'lodi_comtrs'))
    
    
    
    lodi_vineyards.to_file(os.path.join('intermed_data', 'lodi_vineyards'))


    source_dir = os.path.join('source_data', 'Lodi_CDL')
    dst_dir = os.path.join('intermed_data', 'Lodi_CDL')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    
    acres_grapes = {}
    for i, file in enumerate(os.listdir(source_dir)):
        fp = os.path.join(source_dir, file,)
                
        year = file[4:8]
        with rio.open(fp) as r:
            acres_grapes[int(year)] = np.where(r.read(1)==69, 1, 0).sum()*900*.0002477105
            
    pd.DataFrame(acres_grapes.items(), columns = ['years', 'total_acres_grapes']).to_csv(os.path.join('intermed_data', 'lodi_total_grape_acres.csv'))
    
    
    
    #comtrs = gpd.read_file(os.path.join('source_data', 'comtrs'))
    avas = gpd.read_file(os.path.join('source_data', 'CA_avas_shapefile'))
    avas.to_crs(comtrs.crs, inplace =True )

    ava_dissolve = dissolve_all(avas)
    comtrs = intersects(comtrs, ava_dissolve)
    bio_regions = gpd.read_file(os.path.join('source_data', 'cali_bioregions')
                                ).to_crs(comtrs.crs)
    comtrs = max_area_attr_join(comtrs, bio_regions, 'COMTRS', 'INACNAME')
    
    comtrs.to_file(os.path.join('intermed_data', 'ava_comtrs'))
    
    '''
    with rio.open( os.path.join(dst_dir, f'CDL_2007_is_grape.tif')) as r:
        ax = lodi_bounds.plot(alpha = 0)
        show(r, ax = ax)
        
        lodi_vineyards.plot(ax=ax, c= 'r', markersize= 50)
        lodi_counties.geometry.boundary.plot(color = None, 
                                             edgecolor= 'w', ax=ax, linewidth =2)
        #lodi_comtrs.geometry.boundary.plot(color = None,
        #                                  edgecolor = 'w', ax = ax, linewidth = 1)
        plt.show()
    
    ax = lodi_comtrs.plot('acres_grapes')
    lodi_vineyards.plot(ax=ax, c= 'r', markersize= 50)
    '''
    
    
    