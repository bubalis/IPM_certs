#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:44:51 2021

@author: bdube
"""
from org_geocode import get_lat_lon
import pandas as pd
import geopandas as gpd
import os 
import math
import time
from geodata import points_gdf_from_dict
from geopy.adapters import GeocoderUnavailable

os.chdir('spatial')

#%%
comtrs = gpd.read_file(os.path.join('intermed_data', 'ava_comtrs'))


orig_crs = comtrs.crs
comtrs.to_crs('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', inplace = True)

comtrs['TRS'] = comtrs['COMTRS'].apply(lambda x: x[-8:])


#%%
sip_df = pd.read_csv(os.path.join('source_data', 'sip_data_tsv.txt'), 
                     sep = '\t', )

sip_df['lat'] = sip_df['lat'].astype(float)
sip_df['lon'] = sip_df['lon'].astype(float)

lats = []
lons = []
sources = []
for i, row in sip_df.iterrows():
    
    if not math.isnan(row['lat']):
        lats.append(row['lat'])
        lons.append(row['lon'])
        sources.append(row['source'])
    elif type(row['trs']) ==str:
        coords = comtrs[
            (comtrs['TRS'] ==  row['trs']) & 
            (comtrs['NAME']=='Monterey' )].iloc[0].geometry.centroid.coords
        lats.append ( coords[0][1])
        lons.append( coords[0][0])
        sources.append('monterey ranch index')
    elif  type(row['address'])==str:
        print(row['address'])
        while True:
            try:
                lat, lon, source = get_lat_lon(row['address'])
                break
            except GeocoderUnavailable:
                time.sleep(500)
                continue
                
        lats.append(lat)
        lons.append(lon)
        sources.append(source)
        time.sleep(10)
        print(lat, lon)
    else:
        lats.append(row['lat'])
        lons.append(row['lon'])
        sources.append(row['source'])
            
sip_df['lat'] = lats 
sip_df['lon'] = lons
sip_df['source'] = sources               
#%%
sip_shp = gpd.GeoDataFrame(sip_df, 
                           geometry = gpd.points_from_xy(sip_df['lon'], 
                                                         sip_df['lat']),
                crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
#%%
sip_shp.to_crs(orig_crs, inplace = True)

sip_shp.to_file(os.path.join('intermed_data', 'sip_farms'))