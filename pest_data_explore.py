#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:44:50 2021

@author: bdube
"""
import pandas as pd
import os
import numpy as np
os.chdir('spatial')
#%%

df = pd.read_csv(os.path.join('source_data', 'PIP_recs.txt'), sep ='\t')

#PEAS = pd.read_csv(os.path.join('source_data', 'PEAS_toxicity_model.csv'))
#%%

df['PRODUCT_NAME']= df['PRODUCT_NAME'].str.lower()
df['CHEMNAME'] =  df['CHEMNAME'].str.lower().apply(lambda x: str(x).replace('-', ' '))


eiq = pd.read_csv(os.path.join('source_data',  'eiq_values.csv'), encoding = 'latin-1')
eiq['Common.Name'] = eiq['Common.Name'].str.lower().replace('-', ' ').apply(lambda x: str(x).replace('-', ' '))
#%%
unrated_biologicals = '''farnesol
nerolidol
garlic
capsicum oleoresin
sawdust
mpelomyces quisqualis'''.split('\n')

subset = df
#[
#    (df['ADJUVANT']=='NO') & df['PRODUCT_NAME'].str.contains(
#    ' bait|gopher| rat | mouse | mice ')==False]



lookup_names = eiq['Common.Name'].unique()

def assign_lookup_name(name):
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
        return None
    

df['LOOKUPNAME'] = df['CHEMNAME'].apply(assign_lookup_name) 
#%%
percents = []
def perc_lbs_included(df):
    incl = df[df['LOOKUPNAME'].astype(bool)]
    return incl['LBS_AI'].sum()/df['LBS_AI'].sum()


for year in df['YEAR'].unique():
    subset = df[df['YEAR']==year]
    percents.append(perc_lbs_included(subset))

plt.plot(percents)
#%%
mis = df[df['LOOKUPNAME'].astype(bool)==False]
gb = mis.groupby('CHEMNAME')['LBS_AI'].sum().sort_values()

perc = gb/gb.sum()

#%%
tox_data = pd.read_csv(os.path.join('source_data', 'Toxdata Table.txt'), 
                      encoding ='latin-1', sep ='\t'
                       )

#%%
def query_toxdata(query):
    return tox_data[tox_data['CHEMICAL'].str.lower().str.contains(query)]