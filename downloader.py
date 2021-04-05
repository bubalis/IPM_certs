#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:47:51 2021

@author: bdube
"""

import imaplib
import os
#import pandas as pd
import requests
import keyring
import email
from bs4 import BeautifulSoup
import re
import zipfile
import pandas as pd

def login_to_email(gmail_user, password):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    # authenticate
    imap.login(gmail_user, password)
    #imap.select('Inbox')
    #imap.select('Utilities')
    return imap

def get_latest_email(imap, search_string, n):
    out = []
    _, data = imap.search(None, search_string)
    for i in range(1, n+1):
        latest_id = data[0].split()[-1*i]
        _, email_data = imap.fetch(latest_id, "(RFC822)")
        out.append(BeautifulSoup(email_data[0][1]))
    return out

gmail_user = 'benjamintdube@gmail.com'
    
    
gmail_password = keyring.get_password('gmail_imap', gmail_user)

imap = login_to_email(gmail_user, gmail_password)
status, messages = imap.select('INBOX')
res, data = imap.search(None, '(SUBJECT "Your Calpip Data")')

#%%
emails = data[0].split()




def parse_email(e):
    res, email_data = imap.fetch(e, "(RFC822)")
    text = BeautifulSoup(email_data[0][1]).text
    match = re.search('(?<=\r\n\r\n)http:.+(?=\r\n\r\n)', text)
    if match:
        return match.group()


links = [parse_email(e) for e in emails]
    
links2 = ['ftp://transfer.cdpr.ca.gov/pub/outgoing/calpip/' + link.split('id=')[1] +'.zip' 
          for link in links if link]

    
    
#%%
import subprocess
names =[]
for i, link in enumerate(links2):
    name = 'download_data_{i}'
    #download_data = requests.get(link)
    
    subprocess.run(['wget',  link], capture_output= True)
    
    
    # with open(name+'.zip', 'wb') as f:
    #    for chunk in download_data.iter_content(chunk_size=128):
    #       f.write(chunk)
    #with zipfile.ZipFile(name+'.zip',"r") as zip_ref:
    #    zip_ref.extractall(name)
    
    #%%
for file in os.listdir():
    if '.zip' ==file[-4:]:
        with zipfile.ZipFile(file,"r") as zip_ref:
           zip_ref.extractall(file[:-4])
           
#%%
import shutil
years = []
for file in os.listdir():
    if '.zip' ==file[-4:]:
        df = pd.read_csv(os.path.join(file[:-4], file[:-4]+'.txt'), sep ='\t', nrows =20)
        year = df['YEAR'].iloc[0]
        del df
        if year in years:
            os.remove(file)
            shutil.rmtree(file[:-4])
        else:
            years.append(year)