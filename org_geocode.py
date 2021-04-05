#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:32:30 2021

@author: bdube

Script for geocoding the addresses of organic farms from the organic integrity database.

"""
import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim, GeocodeFarm
from geopy.adapters import GeocoderUnavailable
from geopy.exc import GeocoderQuotaExceeded
import time
import os
from pest_lookup import fetch_results
import re
import requests
from spatial_utils import dissolve_all


def geocode_from_gmaps(query):
    '''Scrapes Google Maps using requests to retreive lat-lon values.
    This code only works in the North-West hemi-hemi-sphere, because it assumes
    lat is positive and lon is negative.
    '''
    
    url = 'https://www.google.com/maps/place/' + query.replace(' ', '+')
    res = requests.get(url)
    match =  re.search('(?<=center=)[\d.]+%\d\w-[\d.]+', res.text)
    if match:
        match = match.group()
        lat = float(re.search('[\d.]+', match).group())
        lon = float(re.search('-[\d.]+', match).group())
        return lat, lon
    
def geocode_from_google(query):
    res = fetch_results(query, 20)
    search_name = '\+'.join(res[0].split())
    pattern  = re.compile(f"(?<={search_name}/@)"+'[\-\d.]+,[-\d.]+')
    match = pattern.search(res[1])
    if match:
        pieces = match.group().split(',')
        return [float(piece) for piece in pieces]
    else:
        return None
geolocator = Nominatim(user_agent = 'mics')
back_geocoder = GeocodeFarm()   

def get_lat_lon(string):
    '''Try a bunch of different methods to geocode an address to lat - lon.
    If all fail, return None, None'''
    
    if 'nan, nan' in string:
        return None, None, None
    loc = geolocator.geocode(string)
    if loc:
        return loc.latitude, loc.longitude, "nomatim"
    
    print('Trying Google')
    loc = geocode_from_gmaps(string)
    
    if loc:
        return loc[0], loc[1], 'google maps'
    
    loc = geocode_from_google(string.replace(', ', ' '))
    if loc:
        return loc[0], loc[1], 'google'
    
    time.sleep(6)
    
        
            
    print("Trying Geocode Farm")
    loc = back_geocoder.geocode(string)
    if loc:
        return loc.latitude, loc.longitude, 'Geocode Farm'
    else:
        return None, None, None

if __name__ == '__main__':
    os.chdir('spatial')
    
    
    df = pd.read_csv(os.path.join('source_data', 'Organic_Integrity.csv'), 
                     encoding = 'latin-1')
    
    cali = df[(df['Physical Address: State/Province'] == 'California') | 
       (df['Mailing Address: State/Province']=='California')]
    
    #get only farms that are certified to sell organic grapes. 
    cali.dropna(subset = ['Certified Products Under CROPS Scope'], inplace = True)
    gp = cali[cali['Certified Products Under CROPS Scope'].str.lower().str.contains('grapes')]
    
    
    gp['search_address'] = gp.apply(
        lambda row: f'{row["Physical Address: Street 1"]}, {row[" Physical Address: City"]}, California',
        axis = 1)
    
    
    
    
    
    
    
    
    lats = []
    lons = []
    sources = []
    #%%
    for i, row in gp.iterrows():
        for n in range(3):
            try:
                lat, lon, source = get_lat_lon(row['search_address'])
                lats.append(lat)
                lons.append(lon)
                sources.append(source)
                print(row['search_address'])
                print(lat)
                print(source)
                time.sleep(10)
                break
            except GeocoderUnavailable:
                print('Geocoder Unavailable')
                time.sleep(200)
            except GeocoderQuotaExceeded:
                print('GeoCode Farm passed limit')
                #keep trying tomorrow
                time.sleep(60*60*24)
        else:
            
            lats.append(None)
            lons.append(None)
            sources.append(None)
            
    #%%
    gp['lat' ] = lats
    gp['lon'] = lons
    gp['geocode_src'] = sources
    gp.to_csv(os.path.join('intermed_data', 'organic_vineyards.csv'))
    gp.dropna(subset = ['lat'], inplace =True)
    
    gp = gpd.GeoDataFrame(data = gp, geometry = gpd.points_from_xy(gp['lon'], gp['lat'])
                          )
    gp.crs ='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        
    
    counties = dissolve_all(gpd.read_file(os.path.join('source_data', 'ca-county-boundaries', 'CA_Counties')))
    gp.to_crs(counties.crs, inplace =True)
    keepers = gp.geometry.intersects(counties.geometry.iloc[0])
    bad_data = gp[~keepers]
    #%%
    for i, row in bad_data.iterrows():
        res = back_geocoder.geocode(row['search_address'])
        if res:
            row['lat'] = res.latitude
            row['lon'] = res.longitude
            row['source'] = 'geocode_farm'
    gp = pd.concat([gp[keepers], bad_data])        
    
    gp.to_file(os.path.join('intermed_data', 'organic_vineyards_shp'))
#%%
