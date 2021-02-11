#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:35:04 2020

@author: bdube
"""

os.chdir('pesticide_lists')

files_to_exlc=['aliases.txt', 'chemical_ref_nums.txt',
                 'restricted_pesticides.txt', 'chemical_ref_nums2.txt',
                 'manual_fixes.txt', 'man_fixes_nums.txt', 'aliases2.txt', 'all_results.txt']

def search_all(files, name):
    for file in files:
        data=[d.lower() for d in open(file).read().split('\n')]
        if name in data:
            return file
    
def check_item(dic, item):
    if (item in dic) and dic[item]:
        return dic[item][0]!='Not Listed'
    return False

def lookup_partial(dic, item):
    if check_item(dic, item.split()[0]):
        return dic[item.split()[0]]

def updater(dic1, dic2):
    for k, v in dic2.items():
        if check_item(dic2, k):
            if not check_item(dic1, k):
                dic1[k]=v
            else:
                dic1[k]+=v
    return dic1


files=[f for f in os.listdir() if 'txt' in f and f not in files_to_exlc]

dic=updater(dic, new_man_fixes)

to_solve=[k for k,v in dic.items() if is_not_listed(v) and not check_item(dic, k.split()[0]) ]

to_solve=[k for k in to_solve if k not in alias_man_fixes]

to_solve=[k for k in to_solve if search_all(files, k)]

out={}
for t in to_solve:
    print(t)
    out[t]=full_searcher(t, False)
out={k:v for k,v in out.items() if v}
out={k: v[0][1] for k,v in out.items()}
dic.update(out)
dic['heptachlor']=['76-44-8']
with open('ref_nums.txt', 'w+') as file:
    print(json.dumps(dic), file=file)
    #%%
    
errors=[e for e in errors if e[0]]
out={}
for e in errors:
    res=full_searcher(e[0], False)
    out[e[0]]=res
out={k.lower():v for k,v in out.items() if v}
out={k:v[0][1] for k,v in out.items()}
results.update(out)
#%%
os.chdir('pesticide_lists')

os.getcwd()
with open('ref_nums.txt', 'w+') as file:
    print(json.dumps(dic), file=file)