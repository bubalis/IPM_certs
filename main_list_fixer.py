#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:15:48 2020

@author: bdube

Fix the primary pesticide list of restricted chemicals. 
"""
import os
import pandas as pd
import json
import re
from pest_lookup import full_searcher
#%%

def ecoapple_fix(string):
    '''Fix for combining the Red Tomato Eco with Eco-Apple as one Cert'''
    return string_replacer(string, {'Red Tomato Eco':"Red Tomato Eco Apple"})

def string_replacer(string, mapping):
    '''Replace the string by looking it up in the mapping.
    If string is not in mapping, return the string'''
    response=mapping.get(string)
    if response:
        return response
    else:
        return string


def data_loader():
    df=pd.read_csv(os.path.join('data','all_data.csv'))
    df.fillna('', inplace=True)
    df=df[df['CertName']!='Eco Apple Stonefruit']
    df['CertName']=df['CertName'].apply(ecoapple_fix)
    return df

def ensure_list(item):
    if type(item)==list:
        return item
    else:
        return [item]



#%%


def clip_alias(string):
    '''Clip off an alias of a chemical, by removing all text in parentheses.'''
    if '(' in string:
        return re.split(r'\(.*\)', string)[0].strip()
    else:
        return string
#%%
def CAS_getter(dic, name):
    '''Retrieve the CAS number of name from dic.'''
    
    res = dic.get(name.strip().lower())
    if not res:
        res = dic.get(clip_alias(name.strip()).lower())
    return res
    
def string_fixer(string):
    return string.replace('\xa0', ' ')

def load_ref_num_dict(name):
    dic=json.loads(open(os.path.join('pesticide_lists', name)).read())
    dic={k.lower(): ensure_list(v) for k, v in dic.items()}
    
    
    fixes=json.loads(open(os.path.join('pesticide_lists', 'man_fixes_nums.txt')).read())
    dic.update(fixes)
    return dic

if __name__=='__main__':
    rp=pd.read_csv(os.path.join('pesticide_lists', 'restricted_pesticides.txt'), sep='\t')

    dic=load_ref_num_dict('ref_nums.txt')


    new_data=[]
    for i,line in rp.iterrows():
        cas=line['CAS Number']
        chem_name=string_fixer(line["Chemical Name"])
        '''if 'various' in cas:
            start_data=[]
        elif ',' in cas:
            start_data=cas.split(',')
        else:
            start_data=[cas]'''
        start_data=[]
        response=CAS_getter(dic, chem_name)
        
        if response:
            cas_nums=response+start_data
        else:
            cas_nums=start_data
            
        if not cas_nums:
            for name in chem_name.split(' '):
                print(name)
                response=CAS_getter(dic, name)
                if response:
                    print(response)
                    cas_nums+=response
           
        new_data+=[{**line.to_dict(), **{'CAS Number': cas}} for cas in cas_nums]      
    rp2=pd.DataFrame(new_data)
    rp2.to_csv(os.path.join('pesticide_lists', 'treaty_lists.txt'))
    
    