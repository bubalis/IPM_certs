#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:58:02 2021

@author: bdube
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
import requests
import json
from requests.exceptions import SSLError
from requests.exceptions import ConnectionError
import time
from contextlib import contextmanager

@contextmanager
def progress_saver(fp):
    if os.path.exists(fp):
        
        di = json.loads(open(fp).read())
    else:
        dic = {'data': [], 'start_index' : None}
        
    start_index = di['start_index']
    data = di['data']
    try: 
        yield start_index, data
    finally:
        with open (fp, 'w+') as f:
            print(json.dumps({'data': data, 'start_index': start_index}), file = f)


def check_connection():
    res = requests.get('https://ofmpub.epa.gov/apex/pesticides/pplstxt/10951-19')
    if res.status_code == 500:
        raise
        


os.chdir('spatial')
dfs = []
for substring in ['SEC24C-ACTIVE', 'SEC24C-CANCELLED',
                  'SEC3-ACTIVE', 'SEC3-CANCELLED']:
    fp = os.path.join('source_data', 'PPIS-XML', 
                      f'PPIS-{substring}-03-22-2021.xml')
    xml_data = open(fp, 'r', encoding = 'latin-1').read()  # Read file
    root = ET.XML(xml_data)  # Parse XML
#
    data = []
    cols = []
    for i, child in enumerate(root):
        line = [subchild.text for subchild in child]
        line[12] = [i[0].text for i in child[12]]
        line[13] = [i[1].text for i in child[13]]
        line[14] = ', '.join([i[1].text for i in child[14]])
        line[17] =', '.join( [i[1].text for i in child[17]])
        data.append(line)
        #cols.append(child.tag)
    
    df = pd.DataFrame(data)  # Write in DF and
    df.columns = [c.tag for c in child]
    df['ACTIVE'] = 'CANCELLED' not in substring
    dfs.append(df)
    

df = pd.concat(dfs)
#%%
for t in ['INSECTICIDE', 'HERBICIDE', "FUNGICIDE", 'FUMIGANT']:

    df[t] = df['TYPESLIST'].str.contains(t) | df['PRODUCTNAME'].str.contains(t)
    
apps = pd.read_csv('/mnt/c/Users/benja/Cert_standards/spatial/source_data/PIP_recs.txt', sep ='\t')
apps.drop_duplicates(subset = ['REG_NO'], inplace = True)    

#%%
apps['EPA_regno'] = apps['REG_NO'].apply(lambda x: '-'.join(str(x).split('-')[:2]))
apps['copy_index'] = apps.index
m = apps.merge(df, left_on = 'EPA_regno', right_on = 'EPAREGISTRATIONNUMBER').set_index('copy_index')

m['real_reg_no'] = m['EPAREGISTRATIONNUMBER']

missing = apps.drop(m.index)

#%%
def parse_PPLS_json(res):
    try:
        return json.loads(res.text)['items'][0]['eparegno']
    except IndexError:
        return None
    
def get_real_regno(fake_regno):
    i = 0
    while i<10:
        try:
            res = requests.get('https://ofmpub.epa.gov/apex/pesticides/ppls/' + fake_regno)
        except SSLError:
            print('SSLError')
            time.sleep(50)
            i+=1
        if res.status_code == 200:
            return parse_PPLS_json(res)
        elif res.status_code == 500:
            check_connection()
        else:
            time.sleep(25)
            i+=1
        
def regno_from_name(name):
    
    pieces = name.split()
    if len(pieces) == 1:
        res = requests.get( 'https://ofmpub.epa.gov/apex/pesticides/pplstxt/' + name)
        if res.status_code == 200:
            result = parse_PPLS_json(res)
            if result:
                return result
    while len(pieces)>=2:
        search = ' '.join(pieces)
        res = requests.get( 'https://ofmpub.epa.gov/apex/pesticides/pplstxt/' + search)
        if res.status_code == 200:
            result = parse_PPLS_json(res)
            if result:
                return result
        _ = pieces.pop()
    
real_reg_nos = []


#%%
fp = 'epa_pesticide_nums.txt'
if not os.path.exists(fp):
    start_index = missing.index[0]
    real_reg_nos = []
else:
    di = json.loads(open(fp).read())
    real_reg_nos = di['data']
    start_index = di['start_index']
    
try:
    for i, row in missing.loc[start_index:].iterrows():
        n = 0
        while n<5:
            try:
                res = get_real_regno(row['EPA_regno'])
                if not res:
                    res = regno_from_name(row['PRODUCT_NAME'])
                print(f"{row['EPA_regno']}:   {res}")
                real_reg_nos.append(res)
                break
            except ConnectionError:
                time.sleep(500)
                n+=1
except:
    with open (fp, 'w+') as file:
         print(json.dumps(dict(data = real_reg_nos, start_index = i)),
                          file = file)
    raise
    
            
with open(os.path.join('intermed_data', 'regnos.txt'), 'w+') as f:
    print('\n'.join([str(r) for r in real_reg_nos]), file = f)

missing['real_reg_no'] = real_reg_nos




m2 = missing.dropna(subset =['real_reg_no']).merge(df, left_on = 'real_reg_no', 
                                         right_on = 'EPAREGISTRATIONNUMBER').set_index('copy_index')

missing = missing.drop(m2.index)



m3 = missing.merge(df.drop_duplicates(subset= ['PRODUCTNAME']), 
                   left_on = 'PRODUCT_NAME', right_on = 'PRODUCTNAME').set_index('copy_index')

m3['real_reg_no'] = m3['EPAREGISTRATIONNUMBER']

missing =     missing.drop(m3.index)

missing['id_col2'] = missing['PRODUCT_NAME'].str.replace('[^\w]|\d', '')
missing['MFG_no'] = missing['REG_NO'].apply(lambda x:str(x).split('-')[0] )
df['MFG_no'] = df['EPAREGISTRATIONNUMBER'].apply(lambda x:str(x).split('-')[0])
df['id_col2'] = df['PRODUCTNAME'].str.replace('[^\w]|\d', '')

m4 = missing.merge(df, on = ['MFG_no', 'id_col2']).set_index('copy_index')
m4['real_reg_no'] = m4['EPAREGISTRATIONNUMBER']

missing = missing.drop(m4.index)

found = pd.concat([m, m2, m3, m4])

reg_nos = apps[['REG_NO']].merge(found[['REG_NO', 'real_reg_no']], 
                                 on = 'REG_NO', how = 'left')

reg_nos.to_csv(os.path.join('intermed_data', 'registration_num_translation_key.csv'))
