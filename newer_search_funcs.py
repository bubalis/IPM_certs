#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:04:14 2020

@author: bdube
"""
from contextlib import contextmanager
import json
import os
from pest_lookup import full_searcher
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time

@contextmanager
def progress_saver(*filepaths):
    data_sets=[]
    for fp in filepaths:
        if os.path.exists(fp):
            data_sets.append(json.loads(open(fp).read()))
    else:
        data_sets.append({})
    print(data_sets)
    try:
        yield data_sets
    finally:
        for fp, ds in zip(filepaths, data_sets):
            with open(fp, 'w+') as f:
                print(json.dumps(ds), file=f)
                
def other_name_finder(browser):
    text=browser.find_element_by_xpath("html").text
    results=[]
    for string in text.split('OTHER NAMES FOR')[1].split('\n')[1]:
        if not any([text in string for text in ['Code', 'CAS Number']]):
            results.append(string.strip())
    return results
    
def backup_searcher(name, browser):
    browser=webdriver.Firefox()
    url=f'https://www.pesticideinfo.org/search-result?page=1&sort=Chem_Name&order=ASC&search={name}&type=chemical'
    browser.get(url)
    time.sleep(5)
    elems=browser.find_elements_by_xpath("//a[@href]")
    elems=[e for e in elems if name.lower() in e.text.lower()]
    results=[]
    other_names=[]
    for e in elems:
        open_new_window(browser, e)
    time.sleep(len(elems))
    for i in range(len(elems)):
        print(i)
        browser.switch_to_window(browser.window_handles[(i+1)*-1])
        time.sleep(5)
        
        text=browser.find_element_by_xpath("html").text
        new_cas=find_cas(text)
        results.append(new_cas)
        if new_cas:
            other_names+=other_name_finder(browser)
        time.sleep(5)
    #return results, other_names
def find_cas(text):
    '''FIll in lAter!'''
    pass


def open_new_window(browser, element):
    ActionChains(browser).key_down(Keys.CONTROL).click(element).key_up(Keys.CONTROL).perform()
                
with progress_saver('all_results.txt', 'aliases2.txt') as (all_results, all_other_names):
    
    
    for name in [n for n in names if n not in all_results]:
        results, other_names=backup_searcher(name)
        all_results[name]=[r for r in results if r]
        all_other_names[name]=[o for o in other_names if o]