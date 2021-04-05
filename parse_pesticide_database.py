#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:53:47 2021

@author: bdube
"""

import pandas as pd
import os
import numpy as np

def product_parser(string, instructions):
    pass

def line_parser(string):
    pieces = string.split()
    end = int(pieces[-1])
    start = int(pieces[-2])
    column_name = pieces[0]
    return start-1, end, column_name
#%%
instructions = '''PRODNO NUMBER 7 1 7
MFG_FIRMNO NUMBER 10 8 17
REG_FIRMNO NUMBER 10 18 27
LABEL_SEQ_NO NUMBER 5 28 32
REVISION_NO CHAR 2 33 34
FUT_FIRMNO NUMBER 10 35 44
PRODSTAT_IND CHAR 1 45 45
PRODUCT_NAME VARCHAR2 100 46 145
SHOW_REGNO VARCHAR2 24 146 169
AER_GRND_IND CHAR 1 170 170
AGRICCOM_SW CHAR 1 171 171
CONFID_SW CHAR 1 172 172
DENSITY NUMBER 7 3 173 179
FORMULA_CD CHAR 2 180 181
FULL_EXP_DT DATE 11 182 192
FULL_ISS_DT DATE 11 193 203
FUMIGANT_SW CHAR 1 204 204
GEN_PEST_IND CHAR 1 205 205
LASTUP_DT DATE 11 206 216
MFG_REF_SW CHAR 1 217 217
PROD_INAC_DT DATE 11 218 228
REG_DT DATE 11 229 239
REG_TYPE_IND CHAR 1 240 240
RODENT_SW CHAR 1 241 241
SIGNLWRD_IND NUMBER 1 242 242
SOILAPPL_SW CHAR 1 243 243
SPECGRAV_SW CHAR 1 244 244
SPEC_GRAVITY NUMBER 8 4 245 252
CONDREG_SW CHAR 1 253 253
VAR2_SW CHAR 1 254 254'''.split('\n')
#%%


def parse_instructions(instructions):
    pieces = [i.split() for i in instructions]
    col_names = [i[0] for i in pieces]
    ends = [int(i[-1]) for i in pieces]
    starts = [int(i[-2])-1 for i in pieces]
    types =  [i[1] for i in pieces]
    return col_names, starts, ends, types

def parse_line(line, starts, ends):
    return [line[s:e] for s, e in zip(starts, ends)]

def check_is_float(series):
    return series.str.contains('.').sum() > 0       
        
def parse_data(fp, instructions, encoding = 'latin_1'):
    col_names, starts, ends, types = parse_instructions(instructions)
    
    with open(fp, encoding = encoding) as f:
        data = [parse_line(line, starts, ends) for line in f.read().split('\n')]
    numerics = [col_names[i] for i, t in enumerate(types) if t == 'NUMBER']
    df = pd.DataFrame(data, columns = col_names)
    for col in numerics:
        if check_is_float(df[col]):
            df[col] = df[col].apply(float_0)
        else:
            df[col] = df[col].astype(int)
    return df
  

def float_0(string):
    if string.strip():
        return float(string)
        
    else:
        return 0

main_df = parse_data(os.path.join('source_data', 'productdb', 'product.dat'), 
                     instructions, encoding = 'latin_1')

s_word_instr = '''SIGNLWRD_IND CHAR 1 1 1
SIGNLWRD_DSC VARCHAR2 50 2 51'''.split('\n')

s_word = parse_data(os.path.join('source_data', 'productdb', 'signal_word.dat'),
                    s_word_instr)

s_word['SIGNLWRD_IND'] = s_word['SIGNLWRD_IND'].apply(lambda x: int(x) if x.strip() else np.nan)
main_df = main_df.merge(s_word, on = 'SIGNLWRD_IND',  how = 'left')

main_df.to_csv(os.path.join('intermed_data', 'pesticide_products.csv'))

type_instr = '''PRODNO NUMBER 7 1 7
TYPEPEST_CD CHAR 2 8 9'''.split('\n')


type_df = parse_data(os.path.join('source_data', 'productdb', 'prod_type_pesticide.dat'),
                     type_instr)

type_tab_instr = '''TYPEPEST_CD CHAR 2 1 2
TYPEPEST_CAT VARCHAR2 50 3 52'''.split('\n')

type_tab =  parse_data(os.path.join('source_data', 'productdb', 'type_pesticide.dat'),
                     type_tab_instr)

type_df = type_df.merge(type_tab, on = 'TYPEPEST_CD')[['PRODNO', 'TYPEPEST_CAT']]


type_df.to_csv(os.path.join('intermed_data', 'pesticide_types.csv'))


haz_instr = '''PRODNO NUMBER 7 1 7
ENVHZRD_CD CHAR 2 8 9'''.split('\n')


haz_df = parse_data(os.path.join('source_data', 'productdb', 'prod_env_hazard.dat'),
                     haz_instr)


haz_tab_instr = '''ENVHZRD_CD CHAR 2 1 2
ENVHZRD_DSC VARCHAR2 50 3 52'''.split('\n')

haz_tab = parse_data(os.path.join('source_data', 'productdb', 'env_hazard.dat'),
                     haz_tab_instr)

haz_df = haz_df.merge(haz_tab, on = 'ENVHZRD_CD')[['PRODNO', 'ENVHZRD_DSC']]
haz_df.to_csv(os.path.join('intermed_data', 'haz_table.csv'))






h_haz_instr = '''PRODNO NUMBER 7 1 7
HLHZRD_CD CHAR 2 8 9'''.split('\n')


h_haz = parse_data(os.path.join('source_data', 'productdb', 'prod_health_hazard.dat'),
                    h_haz_instr)

h_tab_haz_instr = '''HLHZRD_CD CHAR 2 1 2
HLHZRD_DSC VARCHAR2 50 3 52'''.split('\n')

h_haz_tab = parse_data(os.path.join('source_data', 'productdb', 'health_hazard.dat'),
                       h_tab_haz_instr)

h_haz = h_haz.merge(h_haz_tab, on = 'HLHZRD_CD')[['PRODNO', 'HLHZRD_DSC']]
h_haz.to_csv(os.path.join('intermed_data', 'health_haz_data.csv'))

