#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:06:05 2021

@author: bdube
"""
#%%
import os
import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import SSLError
import re
import json
from bs4 import BeautifulSoup
from pest_lookup import fetch_results
from geopy.geocoders import Nominatim, GeocodeFarm
import org_geocode
import time


#geolocator = Nominatim(user_agent = 'mics')
#backup_geocoder = GeocodeFarm()







#%%
def parse_to_dict(item):
    out = {}
    lines = re.split('[\t\n]+', item)
    
    for line in lines:
        pieces = line.split(':')
        
        try:
            out[pieces[0]] =pieces[1].strip(',\] ')
        except IndexError:
            continue
    name = out['title'].strip("'").replace('&amp;', '&')
    return name, out



def strip_characters(string):
    'Return a string with only commas, spaces, letters and numbers'
    return re.sub('[^\w\s\d,]', '', string)

#%%
def get_sip_loc_data(res):
    text = res.text.split('var Locations')[1]
    
    
    
    
    t = re.search(r'\[[\s\S]*?\]', text).group()
    t = t.strip('\t\n\[\{')
    items = re.split('\n\t},\n\t{\n', t)
    data = {}
    for item in items:
        res = parse_to_dict(item)
        data[res[0]] =res[1]
    return data

def address_from_website(url):
    try:
        res = requests.get(url)
    except SSLError:
        return None
    
    text = BeautifulSoup(res.text).text
    m = re.search('(?<=\n)\d\d.{30,50}CA 9\d\d\d\d', 
                  text, flags = re.DOTALL)
    if m:
        return re.sub('[^\w^\d^\s]', '',  m.group())
    
    
def get_sip_data():
    res = requests.get('https://www.sipcertified.org/find-wines/')
    
    
    loc_data = get_sip_loc_data(res)
    t2 = res.text.split('<div id="members')[1]
    soup = BeautifulSoup(t2)
    
    divs = soup.find_all('div')
    
    
    vinyards = {}
    for item in divs:
        dic = {}
        string = str(item)
        if 'Vineyard Icon' in string:
            text = item.text
            pieces = [i for i in text.split('\n') if i]
            name =pieces[0]
            county = pieces[1]
            year_match = re.search('(?<=Certified since )\d\d\d\d', text)
            if 'Website' in string:
                url = item.findAll('a')[0].get('href')
                print(url)
                dic['url'] = url
                address = address_from_website(url)
                print(address)
                if address:
                    address = ' '.join(parse_mashed_string(address))
                    dic['address'] = address
                    lat, lon, source = org_geocode.get_lat_lon(address)
                    time.sleep(5)
                    dic['lat'] = lat
                    dic['lon'] = lon
                    dic['source'] = source
                    
            if 'Certified' not in county:
                dic['county'] = county
            else:
                dic['county'] = None
            
            if year_match:
                 dic['Year Certified']= year_match.group()
            else:
                 dic['Year Certified'] = None
            vinyards[name] = dic
    
    for k, v in vinyards.items():
        matches = [n for n in loc_data if n in k]
        if matches:
            match = matches[0]
            v['lon'] = loc_data[match]['lon']
            v['lat'] = loc_data[match]['lat']
            v['source'] = 'Sip Website Map'
        vinyards[k] = v  
    
    
    for k, v in vinyards.items():
        if 'lon' in v or 'address' in v:
            pass
        else:
            print(k)
            if v['county']:
                search_name = k+' '+ v['county']
            else:
                search_name = k
            result = scrape_from_google(search_name)
            if result:
                print(result)
                v['address'] = result
                lat, lon, source = org_geocode.get_lat_lon(strip_characters(result))
                v['lat'] = lat
                v['lon'] = lon
                v['source'] = source
            else:
                v['address'] = None
            time.sleep(8)
    
    return vinyards

def scrape_from_google(k):
    for extras in (' California vineyard', ' California Winery'):
        result = search_term(k, extras)
        if result:
            return result
    
def search_term(word, extras):
    _, res = fetch_results(word+ ' ' + extras, 5)
    soup = BeautifulSoup(res)
    m = re.search('(?<=Address: ).*? \d\d\d\d\d', soup.text)
    if m:
        if 'CA' in m.group():
            return m.group()


'''
def get_lat_lon(address, geolocator):
    loc = geolocator.geocode(address)
    if loc:
        return dict(address = address, lat = loc.latitude, lon = loc.longitude)
    else:
        loc = backup_geocoder.geocode(address)
    if loc:
        return dict(address = address, lat = loc.latitude, lon = loc.longitude)
    return dict(address = address)
'''

def clean_zip(string):
    return re.sub('(?<=\d)\d\d\d\.\d\d\d\.\d\d\d\d', '', string)

def parse_mashed_string(string):
    '''Parse a string where spaces have been removed at line breaks.'''
    
    pattern = '|'.join(['[a-z][A-Z]', '[a-z]\d', '\d[A-Z]', '(?<=[a-zA-Z])\.\d',
                        '(?<=LL)C\d'])
    
    split_patterns =re.findall(pattern, string)
    splits = re.split(pattern, string)    
    for i, s in enumerate(splits):
        try:
            if i!=0:
                s = split_patterns[i-1][1]+ s
            if i != len(split_patterns):
                s += split_patterns[i][0]
            splits[i] = s   
        except:
            print(string)
            print(split_patterns)
            raise
    return [clean_zip(line) for line in  splits]


#%%

def get_lodi_data():
    #downloading data for the LODI farms    
    res = requests.get('https://www.lodigrowers.com/growers/')
    soup = BeautifulSoup(res.text)
    divs = soup.find_all('div')
    text = divs[13].text.split('Below is a list of LODI RULES growers. Are you a LODI RULES grower who needs to add or update your contact information? Call the Lodi Winegrape Commission at 209.367.4727 and ask to be updated in the LODI RULES Certified Vineyard list.')[1]
    
    
    
    
    lodi_data = [parse_mashed_string(item) for item in text.split('\n') if item.strip()]
    
    lodi_addresses = {item[0]: item[1]+ ', '+ item[2] 
                      for item in [i for i in lodi_data if len(i)>2] if 'PO Box' not in item[1]}
    
    lodi_out = {}
    for k, address in lodi_addresses.items():
        
        
        lat, lon = org_geocode.get_lat_lon(address)
        lodi_out[k] = {'Address': address, 'lat': lat, lon: 'lon'}
        time.sleep(1.5)
    return lodi_out
    
    
    
def remove_zip(string):
    return re.sub(' \d\d\d\d\d$', '', string)


def add_leading_0s(n, _len = 2):
    n = str(n)
    to_add = _len - len(n)
    if to_add>0:
        
        return '0' * to_add + n
    else:
        return n

def parse_COMTRS(row):
    co = row['alpha_code']
    M = row['Meridian'][0]
    T = row['Township'][1:]
    R = row['Range'][1:]
    S = add_leading_0s(str(row['Section']))
    return co+M+T+R+S

def add_comtrs(comtrs_gdf, counties_gdf):
    pass


def points_gdf_from_dict(data, dst_crs, extra_keys=[]):
    '''From a dict of dicts with lat lon data, make a geodataframe of points.'''
    
    p = gpd.GeoDataFrame(data = {**{'Name': [key for key in data]},
                                 **{k: [v[k] for v in data.values()] for k in extra_keys}},                
                                 geometry= gpd.points_from_xy([float(entry['lon']) for entry in data.values()],
                             [float(entry['lat']) for entry in data.values()]))
    
    
    p.crs ='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    p.to_crs(dst_crs,inplace=True)
    return p
    
    
#%%
if __name__=='__main__':
    os.chdir('spatial')
    
    lodi_path = os.path.join('source_data', 'lodi_data.txt') 
    sip_path = os.path.join('source_data', 'sip_data.txt')
    sip_data = get_sip_data()
    with open(sip_path, 'w+') as f:
            print(json.dumps(sip_data), file = f)
    sip_df = pd.DataFrame(sip_data)
    sip_df.to_csv(os.path.join('source_data', 'sip_data_tsv.txt'), sep ='\t')
    
    if not os.path.exists(lodi_path):
        lodi_data = get_lodi_data()
        with open(lodi_path, 'w+') as f:
            print(json.dumps(lodi_data), file = f)
    else:
        lodi_data = json.loads(open(lodi_path).read())
    
    if not os.path.exists(sip_path):
        sip_data = get_sip_data()
        with open(sip_path, 'w+') as f:
            print(json.dumps(sip_data), file = f)
    else:
        sip_data = json.loads(open(sip_path).read())
    
    counties = gpd.read_file(os.path.join('source_data','ca-county-boundaries', 'CA_Counties'))    
    comtrs_path = os.path.join('source_data', 'comtrs')
    if not os.path.exists(comtrs_path):
        gdf = gpd.read_file(os.path.join('source_data', 
                                         'Public_Land_Survey_System_(PLSS)%3A_Sections'))
        
        
        counties.sort_values('NAME', inplace = True)
        counties['alpha_code'] = [add_leading_0s(n+1) for n in range(counties.shape[0])]
        merge = gpd.overlay(gdf, counties)
    
    
    
        merge['COMTRS'] = merge.apply(parse_COMTRS, axis =1 )
        merge.to_file(comtrs_path)
    
    '''
    p = points_gdf_from_dict(lodi_data, counties.crs)
    p = gpd.clip(p, counties)
    
    p2 = points_gdf_from_dict({k:v for k, v in sip_data.items() if 'lon' in v}, 
                              counties.crs, extra_keys = ['Year Certified'])
        
    p2 = gpd.clip(p2, counties)
    
    ax = counties.plot(color ='w', edgecolor = 'k')
    p.plot(alpha = .5, ax= ax)
    p2.plot(ax=ax, alpha =.5)
    ax.axis('off')
    
    
    p.to_file(os.path.join('intermed_data', 'lodi_vinyards'))
    p2.to_file(os.path.join('intermed_data', 'sip_vinyards'))
    '''