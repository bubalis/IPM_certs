#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:05:12 2020

@author: bdube
"""
import subprocess
import re
import wikipedia
import pandas as pd
from bs4 import BeautifulSoup
import os
import json
import csv
import requests
from wikipedia import PageError, DisambiguationError
import time



def fetch_results(search_term, number_results, language_code='en'):
    USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    assert isinstance(search_term, str), 'Search term must be a string'
    assert isinstance(number_results, int), 'Number of results must be an integer'
    escaped_search_term = search_term.replace(' ', '+')
 
    google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results, language_code)
    response = requests.get(google_url, headers=USER_AGENT)
    response.raise_for_status()
    print(google_url)
    return search_term, response.text

def parse_results(html, keyword):
    soup = BeautifulSoup(html, 'html.parser')
 
    found_results = []
    rank = 1
    result_block = soup.find_all('div', attrs={'class': 'g'})
    for result in result_block:
 
        link = result.find('a', href=True)
        title = result.find('h3')
        description = result.find('span', attrs={'class': 'st'})
        if link and title:
            link = link['href']
            title = title.get_text()
            if description:
                description = description.get_text()
            if link != '#':
                found_results.append({'keyword': keyword, 'rank': rank, 'title': title, 'description': description, 'link': link})
                rank += 1
    return found_results
def scrape_google(search_term, number_results, language_code='en'):
    try:
        keyword, html = fetch_results(search_term, number_results, language_code)
        results = parse_results(html, keyword)
        return results
    except AssertionError:
        raise Exception("Incorrect arguments parsed to function")
    except requests.HTTPError:
        raise Exception("You appear to have been blocked by Google")
    except requests.RequestException:
        raise Exception("Appears to be an issue with your connection")

def webchem_search(name):
    r_path= '/home/bdube/miniconda3/envs/IPM/bin/Rscript'
    r_script="/mnt/c/Users/benja/Cert_standards/pesticide_lists/CAS_Lookup.R"
    args=[r_path, r_script, name]
    i=0
    while i<5:
        try:
            response=subprocess.check_output(args)
            return response_parser(response)
        except:
            print('Connection Refused... Waiting it out')
            time.sleep(200)
            i+=1
    


def response_parser(x):
    as_list=re.split(r'\[1\]|\\n', str(x))
    as_list=[a.replace('"', '').replace("'", '' ).strip() for a in as_list]
    
    tup=(as_list[1], [a for a in as_list[2:] if 'Failed' not in a and a])
    valid= tup[1] and tup[1][0]!='NA'
    return valid, tup
             


def wikiSearcher(chem_name):
    time.sleep(2)
    responses=wikipedia.search(chem_name)
    for response in responses:
        try:
            soup= BeautifulSoup(wikipedia.page(response).html())
        
            res_block= soup.find_all('div')
            for i, res in enumerate(res_block):
                if "CAS Number" in res.text:
                    match=re.search('[\d\-]{5,}', res_block[i+1].text)
                    if match:
                        return True, (response, match.group().split(' '))
        except (PageError, DisambiguationError, KeyError) :
            continue
    return False, (None, None)



    

def read_csv_list():
    results={}
    with open('restricted_pesticides.txt') as file:
        reader=csv.reader(file, delimiter='\t')
        lines=[line for line in reader]
        for line in lines:
            if line[2] in ['various', ''] and line[1] not in results:
                results=full_searcher(line[1], results)
            elif line[2]:
                results[line[1]]=[line[2]]
    return results


def get_from_chembook(name):

    time.sleep(15)
    responses=scrape_google(name, 25)
    for r in responses:
        if 'www.chemicalbook.com' in r['link']:
            print(r['link'])
            try:
                return True, (name, scrape_chembook(r['link']).split(' '))
            except:
                continue
    else:
        return None, (None)


def scrape_chembook(url):
    response=requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    res_block=soup.find_all('div')
    return check_resblock(res_block)

def check_resblock(res_block):
    for r in res_block:
        if 'CAS' in r.text:
            return re.search('[\d\-]{5,}', r.text).group()


search_functions=[webchem_search, wikiSearcher, get_from_chembook]



def full_searcher(name, pieces=True):
    for search_function in search_functions:
        valid, tup=search_function(name)
        if valid:
            return [tup]
    
    if pieces:
        if len(name.split(' '))>1:
            responses=[]
            for n in [n for n in name.split(' ') if len(n)>3]:
                responses+=full_searcher(n)
            return [r for r in responses if r]
    else:
        return []

def full_search_and_append(name, dic, aliases):
    if name not in dic and name not in aliases:
        responses=full_searcher(name)
        dic[name]=[]
        for tup in responses:
            dic[tup[0]]=tup[1]
            if tup[0]!=name:
                aliases[tup[0]]=name
    return dic, aliases


def replacer(string, mapping):
    response=mapping.get(string.lower())
    if response:
        return response
    else:
        return string.lower()

def unpack_string_list(values):
    if len (values)==1:
        return [v.strip() for v in values[0].split(' ')]

#%%
def data_saver(results, aliases):
    '''Save data from results and aliases.'''
    with open('chemical_ref_nums.txt', 'w+') as file:
        
        print(json.dumps(results), file=file)
    with open('aliases.txt', 'w+') as file:
        print(json.dumps(aliases), file=file)
        
def initial_loader():
    '''Load data from previous partial runs of the script, if it exists.
    Otherwise, return two empty dictionaries.'''
    out=[]
    for f in ('chemical_ref_nums.txt', 'aliases.txt'):
        if f in os.listdir():
            out.append(json.loads(open(f).read()))
        else:
            out.append({})
    return out
        

if __name__=='__main__':
    os.chdir('pesticide_lists')
    
    results, aliases=initial_loader()
    #results=read_csv_list()
    mapping=json.loads(open('manual_fixes.txt').read())
    
    for f in [f for f in os.listdir() if f[-4:]=='.txt']:
        if f in ['aliases.txt', 'chemical_ref_nums.txt',
                 'restricted_pesticides.txt', 'chemical_ref_nums2.txt',
                 'manual_fixes.txt', 'man_fixes_nums.txt']:
            continue
        with open(f, encoding='latin1') as file:
            print(f)
            for line in file.read().split('\n'):
                
                line=replacer(line, mapping)
                if line and line not in results:
                    results, aliases=full_search_and_append(line, results, aliases)
                
    data=pd.read_csv('restricted_pesticides.txt', sep='\t')
    for name in data['Chemical Name'].tolist():
        name=replacer(name, mapping)
        if line and line not in results:
            results, aliases=full_search_and_append(line, results, aliases) 
                    
    results={k:unpack_string_list(v) for k,v in results.items() if v}
    data_saver(results, aliases)
    
    print([k for k,v in results.items() if not v])