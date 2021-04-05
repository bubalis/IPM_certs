#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:10:21 2020

@author: bdube

Read all txt files listing pesticides which are 
banned or restricted by farms within certifications.
Aggregate this data and save it 

"""
from contextlib import contextmanager
from main_list_fixer import data_loader, CAS_getter, load_ref_num_dict
import pandas as pd
import json
import os
from browser_automation_scraper import backup_searcher
#%%
def filter_reqs(df):
    '''Only get requirements that are fully required, eventually.
    Do not include scorecard / optional standards. '''
    return df[df['Required/Core or Improvement'].isin(
        ['ScoreCard', 
        'Scorecard', '', 
        'scorecard',
       'C', 'recommended',
        'Improvement'])==False]


df=data_loader()
ban=filter_reqs(df)
#%%
ban['Prohibited Material'].unique()

#%%
ref_num_dict=load_ref_num_dict('chemical_ref_nums2.txt')
assert ('Tributyltin Compounds'.lower() in ref_num_dict)
assert ('Mercury and its compounds'.lower() in ref_num_dict)

def fix_ref_num_dict(ref_num_dict):
    #ref_num_dict={k:v for k,v in ref_num_dict.items() if all([subv for subv in v])}
    to_del=[]
    for k,v in ref_num_dict.items():
       
        if k=='\\\\\\':
            to_del.append(k)
        elif any([', ' in subv for subv in v]):
            ref_num_dict[k]=[item.strip() for item in v[0].split(',')]
    
    for k in to_del:
        del ref_num_dict[k]
        
    assert not any([any([', ' in subv for subv in v]) for v in ref_num_dict.values()]), [k for k,v in ref_num_dict.items() if any([',' in subv for subv in v])] 
    return ref_num_dict


ref_num_dict=fix_ref_num_dict(ref_num_dict)


assert ('Tributyltin Compounds'.lower() in ref_num_dict)




#%%

exclude=['only use legal pesticides',
         'any pesticides prohibited by law',
         'must only use legal pesticides',
  'must only handle pesticides labelled for pest',
  'obey all laws', 
  'obey restrictions from customers',
  'explosives for wildlife pests', 
  'follows law', ' any pesticides prohibited by law'
         ]

@contextmanager
def cwd(path):
    '''Context manager to change working directory and change it back.'''
    
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

keys_to_simplify=['who', 'rotterdam', 'stockholm']



def simplify_listnames(vals):
    
    
    for k in keys_to_simplify:
        if any([k in v for v in vals]):
            vals=[v for v in vals if k not in v]
            vals.append(k)
            
    return vals





#%%
def load_epa_CAS():
    '''Load CAS #s of all EPA-registered Pesticides'''
    with cwd('Pesticides - Active Ingredients'):
        df=pd.read_csv("Pesticides - Active Ingredients.csv")
        return df['CAS #'].tolist()
    
assert ('Tributyltin Compounds'.lower() in ref_num_dict)
errors=[]


def load_CASlist(file):
    global errors
    chem_names=open(file).read().split('\n')
    cas_nums=[]
    for chem_name in chem_names:  
        if not chem_name.strip():
            continue
        res=CAS_getter(ref_num_dict, chem_name)
        if res:
            cas_nums+=res
        else:
            #results, other_names = backup_searcher(chem_name)
            #if results:
            #    cas_nums+=results
            #else:
                print(chem_name)
                print(chem_name==chem_name.strip())
                errors.append((chem_name, file))
    return [n for n in cas_nums if type(n)==str]


def make_plist():
    with cwd('pesticide_lists'):
        p_list={}
        for key in ['who', 'rotterdam', 'stockholm']:
            p_list[key]=load_CASlist(f'{key}.txt')
        return p_list
        
plist_dic=make_plist()

def load_propre_list(name, list_type, cas_nums_incl=False):
    '''Load the text file listing the cert-specific list of
    banned or restricted pesticides.
    args:
        _name_: certification name
        _list_type_ : "banned" or "restricted" '''
    
    print(name)
    with cwd('pesticide_lists'):
       
        namer=f"{'_'.join(name.split(' '))}_{list_type}.txt".lower()
        print(namer)
        try:
           file=[f for f in os.listdir() if namer in f][0]
        except:
            pass
        
        if '.txt' in file:
            cas_nums=load_CASlist(file)
            
            
        
            
    if any(['\n' in num for num in cas_nums]):
        print (name)
        raise ValueError
    return cas_nums
    


specific_chems=['paraquat', 'carbofuran']
        



#%%%

CAS_to_ignore=load_CASlist(os.path.join('pesticide_lists/irrelevant_pesticides.txt'))
CAS_to_ignore.append('Not Listed')



def get_CAS_list(cert_name, values, list_type):
    '''From the data values for restricted or 
    banned chemicals, create a list of unique CAS numbers.'''
    
    cas_list=[]
    for key in keys_to_simplify:
        if key in values:
            cas_list+=plist_dic[key]
    if 'propreitary list' in values:
        cas_list+=load_propre_list(cert_name, list_type)
        
        #meythl bromide is listed in the montreal protocol
    if any(['montreal' in v for v in values]):
        cas_list+=CAS_getter(ref_num_dict, 'methyl bromide')
    for chem in specific_chems:
        if chem in values:
            res=CAS_getter(ref_num_dict, chem)
            if res:
                cas_list+=res
                
    return set([c for c in cas_list if c not in CAS_to_ignore])
        
def data_saver(data_to_save, string):
    with open(os.path.join('data', f'cert_pest_lists_{string}.txt'), 'w+') as file:
        print(json.dumps(data_to_save), file=file)

def summarize_data(data, string):
    '''Return a dictionary summarizing the counts of banned/restricted
    pesticides in an individual organization's certification.
    the dict has 3 values:
        'num' : total number
        'num_epa_reg' : count of pesticides registered for use by the EPA
        'num_not_epa_reg' count of pesticides registered for use by the EPA
        
    
    '''
    
    epa_cas=load_epa_CAS()
    data_to_save={}
    data_to_save['num']={k: len(v) for k,v in data.items()}
    data_to_save['num_not_reg_epa']={k: len(
        [c for c in v if c not in epa_cas]) for k,v in data.items() }
    data_to_save['num_epa_reg']={k: len(
        [c for c in v if c in epa_cas]) 
        for k,v in data.items() }
    return data_to_save

#%%
for list_type, column in zip(
        ('banned', 'restricted'),
        ('Prohibited Material','Limited Use Substances')): 
    
    print(list_type)
    by_cert={cert: [i for i in 
                    ban[ban['CertName']==cert][column].unique() if i]
                    for cert in df['CertName'].unique()}
    by_cert={key: [x for y in value for x in y.split(',')]
             for key, value in by_cert.items()}
    
    by_cert={key:[v for v in value if v not in exclude] 
             for key, value in by_cert.items()}
    
    by_cert={key:simplify_listnames(vals) for key, vals in by_cert.items() if key}
    
    
    
    data= {key: get_CAS_list(key, values, list_type) for key, values in by_cert.items()}
    
    
    data_to_save = summarize_data(data, list_type)
    data_to_save=data_saver(data_to_save, list_type)

