#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:14:07 2021

@author: bdube
"""


import pandas as pd
import geopandas as gpd
import os
from spatial_utils import total_area_eq_val,  get_rast_crs, parallelize_on_rows

import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

import numpy as np
import pickle
from functools import partial
import itertools

def try_merge(left, right, chunksize=1000, **kwargs):
    '''Merge two dataframes together, using the standard pd.merge function.
    If this results in a memory error, save the right-side dataframe 
    to a scratch csv and merge in chunks.''' 
    
    try:
        return left.merge(right, **kwargs)
    
    except MemoryError:
        out = []
        right.to_csv('scratch.csv')
        reader = pd.read_csv('scratch.csv', chunksize = chunksize)
        for r in reader:
            out.append(left.merge(r, **kwargs))
            del r
        os.remove('scratch.csv')
        return pd.concat(out)






def filter_data(df, match_col, filter_df, filter_col, filter_val):
    '''Get all values of df where the matching entry for match_col in filter_df
    has a corresponding value for filter_col == filter_val.
    df:              df2:
        a               a   b 
    1   6            1  6  's' 
    2   4            2  6  'f' 
    3   2            3  4  'm'
                     4  4  'r' 
                     5  2  's' 
    filter_data(df, 'a', df2, 'b', 's')
    returns  
       a
    1  6
    3  2
    
    '''
    merged = df.merge(filter_df, on = match_col)
    
    return merged[merged[filter_col] == filter_val].drop(columns = [filter_col])
    

def get_cert_years(shapes_gdf, points_gdf, year_col, year_range, 
                   name_col = 'COMTRS', lead_time = 0, out_name= 'N_organic'):
    '''From a shapes gdf and a points gdf with a year column,
    make a dataframe listing the number of points within each value for name_col
    that are less than or equal to each year. 
    Gives the number of certified farms in each sub-geometry estimated to 
    be present for each year.'''
    
    joined = gpd.sjoin(shapes_gdf, points_gdf)
    out = []
    joined[out_name] = 1
    for year in year_range:
        sub_points = joined[joined[year_col]<=(year-lead_time)]
        num_points = sub_points.groupby(name_col)[[out_name]].sum()
        num_points['YEAR'] = year
        out.append(num_points)
    return pd.concat(out).reset_index()


def make_random_effects(df, col):
    dic = {e: i for i, e in enumerate(np.sort(df[col].unique()))}
    return df[col].replace(dic).values, dic


def collect_neighbors(geometry, gdf, sindex = None, id_col = "COMTRS"):
    '''Get all the neighbors of a geometry from a gdf.
    If id_col is passed, return a list of values for id_col.
    pass a spatial index (sindex) to speed queries. 
    '''
    if sindex:
        indicies =list(sindex.intersection(geometry.bounds))
        sub = gdf.loc[indicies]
    else:
        sub = gdf
    if id_col:
        return sub[~sub['geometry'].disjoint(geometry)][id_col].tolist()
    else:
        return sub[~sub['geometry'].disjoint(geometry)]

def agg_neighbors(geometry, gdf, col, sindex, agg_func = np.sum):
    return agg_func(collect_neighbors(geometry, gdf)[col])






os.chdir('spatial')
#%%
preped_data_path = os.path.join('intermed_data', 'read_for_analysis.csv')

if os.path.exists( preped_data_path):
    data = pd.read_csv(preped_data_path)
else: 
    comtrs = gpd.read_file(os.path.join('intermed_data', 'ava_comtrs'))
    
    org_vineyards = gpd.read_file(os.path.join('intermed_data', 
                                               'organic_vineyards_shp'))
    
    org_vineyards['Year'] = org_vineyards['Effective'].apply(lambda x: 
                                                             int(x.split('-')[-1]))
    
    #remove multiple entries for the same address
    org_vineyards.drop_duplicates(subset = ['search_add'], inplace = True)
    org_vineyards.to_crs(comtrs.crs, inplace = True)
    org_vineyards = org_vineyards[org_vineyards.geometry.is_valid]
    
    
    #load in the lodi data
    lodi_comtrs = gpd.read_file(os.path.join('intermed_data', 'lodi_comtrs'))
    lodi_cert_acres = pd.read_csv(os.path.join('source_data', 'lodi_acres.csv'))
    lodi_grape_acres = pd.read_csv(os.path.join('intermed_data', 'lodi_total_grape_acres.csv'))
    lodi_cert_by_year = lodi_cert_acres.merge(lodi_grape_acres, on = 'Year')
    
    lodi_cert_by_year['perc_certified'] = lodi_cert_by_year['lodi_acres']/lodi_cert_by_year['acres_grapes']
    lodi_cert_dict = dict(zip(lodi_cert_by_year['Year'].tolist(), lodi_cert_by_year['perc_certified'].tolist()))
    
    
    
    
    
    cert_by_year = get_cert_years(comtrs, org_vineyards, 'Year', range(2003, 2020))
    
    del org_vineyards
    
    
    
    #rast_fp = os.path.join('source_data', 'Cali_CDL', 'CDL_2018.tif')
    #rast_crs = get_rast_crs(rast_fp)
    
    acres_grapes = pd.read_csv(os.path.join('intermed_data', 'california_grape_acres.csv')).drop(columns = 'Unnamed: 0')
    acres_grapes = acres_grapes.melt(id_vars = 'COMTRS', 
                                     value_name ='acres_grapes', 
                                     var_name = 'YEAR')
    
    acres_grapes['YEAR'] = acres_grapes['YEAR'].apply(lambda x: int(x.replace('acres_grapes_', '')))
    
    
    
    
    
    
    
    
    #add a column listing the neighbors of each COMTRS
    comtrs['neighbors'] = comtrs['geometry'].apply(partial(collect_neighbors, 
                                        gdf = comtrs, sindex = comtrs.sindex))
    
    
    #%%
    apps = pd.read_csv(os.path.join('source_data', 'PIP_recs.txt'), sep = '\t')
    
    apps = apps[['COMTRS', 'GROWER_ID', 
                 'AREA_PLANTED', 'YEAR', 'LBS_AI', 'PRODNO',
                 ]]
    prod_data = pd.read_csv(os.path.join('intermed_data', 'pesticide_products.csv'))
    
    apps = apps.merge(prod_data[['PRODNO', 'SIGNLWRD_IND']], on = 'PRODNO', how = 'left')
    apps['SIGNLWRD_IND'] = apps['SIGNLWRD_IND'].fillna(5)
    
    del prod_data
    
    apps = apps[apps['YEAR']>1990]
    
    
    
    def agg_from_neighbors(indicies, gdf, agg_col, matcher_col= 'COMTRS'):
        try:
            if matcher_col:
                d = gdf[gdf[matcher_col].isin(indicies)]
                return d[agg_col].sum() 
            else: 
                return gdf.loc[indicies, agg_col].sum()
        except:
            print(indicies)
            raise
    
    
    def vector_neigh_match(gdf, agg_col, index_col = 'neighbors',  subset_col = 'YEAR',
                     matcher_col = 'COMTRS'):
        '''Vectorized function for collecting data from neighbors of each shape. '''
        out = []
        for subset_val  in gdf[subset_col].unique():
            print(subset_val)
            subset = gdf[gdf[subset_col] == subset_val]
            func = partial(agg_from_neighbors, gdf = subset, agg_col = agg_col, matcher_col = matcher_col)
            out.append(subset[index_col].apply(func))
            # gdf.loc[subset.index, new_col_name] = subset[new_col_name]
            
        return pd.concat(out)
    #
    
    cert_by_year = cert_by_year.merge(comtrs[['COMTRS', "neighbors"]], on = 'COMTRS')
    cert_by_year['N_organic_neigh'] = vector_neigh_match(cert_by_year, 'N_organic')
    
    #%%
    cols_to_keep = ['COMTRS',
                    'NAME', 
                    'YEAR', 
                    'acres_grapes', 
                    'neighbors', 
                     'INACNAME'
                                        ]
    
    data = comtrs.merge(acres_grapes, on = ['COMTRS'], how = 'left')
    #data = data[(data['acres_grapes']>20) | (data['acres_grapes'].isna())][cols_to_keep]
    
    data = data[cols_to_keep]
    
    data =  data.merge(cert_by_year, on = ['YEAR', 'COMTRS'], how = 'left') 
    data = try_merge(data, apps, chunksize=1000,  on = ['COMTRS', 'YEAR'], how = 'left')
    
    del apps
    operations = data.groupby(['COMTRS', 'YEAR'])[['GROWER_ID']].agg(lambda x: len(set(x))
                            ).rename(columns={'GROWER_ID': 'num_operations'})
    
    operations = operations.join(comtrs[['COMTRS', "neighbors"]].set_index('COMTRS')).reset_index()
    
    
    
    operations['num_operations_neigh'] = vector_neigh_match(operations, 'num_operations')
    
    
    
    data = data.merge(operations, how = 'inner', on = ['COMTRS', 'YEAR'])
    data['N_organic'] = data['N_organic'].fillna(0)
    data['perc_certified_lodi'] = 0
    data.loc[data['COMTRS'].isin(lodi_comtrs['COMTRS'].tolist()), 'perc_certified_lodi'] = data.loc[data['COMTRS'].isin(
        lodi_comtrs['COMTRS'].tolist()), 'YEAR'].apply(
            lambda x: lodi_cert_dict.get(x, 0))
    
    
    
    
    
    
    #add in indicator for Napa Valley:
    napa_data = pd.read_csv(os.path.join('intermed_data', 'synthetic_napa_data.csv'))
    napa_dict = dict(zip(napa_data['Year'], napa_data['perc_napagreen']))
    
    
    
    data['perc_certified_napa'] = 0
    data.loc[data['NAME'] == 'Napa', 'perc_certified_napa'] = data.loc[
        data['NAME'] == 'Napa', 'YEAR'].apply(lambda x: napa_dict.get(x, 0))
    
        
    
    
    type_df = pd.read_csv(os.path.join('intermed_data', 'pesticide_types.csv'))
    
    
    #data['N_organic_neigh'] = data.apply(lambda x: org_agg_func(x['neighbors'], x['YEAR']), axis =1)
    
    #ops_agg_func =  partial(agg_from_neighbors, gdf = data, agg_col = 'num_operations')
    #data['num_operations_neigh'] = ops_agg_func(data['neighbors'], data['YEAR'])
    
    #data['LBS_PER_AC'] 
    #data['LBS_PER_AC'] = data['LBS_AI']/data['acres_grapes']
    data['N_organic_neigh'].fillna(0, inplace = True)
    data['num_operations_neigh'].fillna(0, inplace = True)
    #data = data.merge(type_df, on = 'PRODNO')
    
    
    data = data.drop(columns = ['neighbors_x', 'neighbors_y', 'neighbors', 
                                ])
    del lodi_comtrs
    del comtrs
    del acres_grapes
    del cert_by_year
    del operations
    data.to_csv(preped_data_path)
#%%

def prep_gb(data, full_data):
    '''Regroup pesticide data aggregated based on COMTRS and 
    make necessary calcuated columns.'''
    
    df = data.groupby(['COMTRS', 'YEAR', 'GROWER_ID'])[['LBS_AI']].sum()
    
    df = df.reset_index().set_index(['COMTRS', "YEAR"])
    
    gb = full_data.groupby(['COMTRS', 'YEAR'])
    df = df.merge(gb[['NAME',  'num_operations', 'N_organic', 'acres_grapes',
                       'num_operations_neigh', 'N_organic_neigh', 'INACNAME', 
                       'perc_certified_lodi', 'perc_certified_napa'
                       ]].agg('first'),
                  how= 'right', left_index= True, right_index = True)
    
    df = df.merge(gb[['AREA_PLANTED']].agg(lambda x: x.unique().sum()),
                  how = 'right', left_index= True, right_index = True)
    
    df['LBS_AI'] = df['LBS_AI'].fillna(0)
    
    df.rename(columns = {'NAME': 'County', 'AREA_PLANTED': 'ACRES_PLANTED'}, inplace =True)
    
    filter_1 = lambda x: x if x<=1 else 1
    #df['LBS_PER_AC_area_from_PUR'] = df['LBS_AI']/df['ACRES_PLANTED']
    #df['LBS_PER_AC_area_from_geodata'] = df['LBS_AI'] / df['acres_grapes']
    df['geodata_available'] = (df['acres_grapes'].isna() == False).astype(int)
    
    
    df['perc_certified_org'] = (df['N_organic'] / df['num_operations']
                                )#.apply(filter_1)
    
    df['perc_certified_neigh_org'] = (df['N_organic_neigh'] / df['num_operations_neigh']
                                      )#.apply(filter_1)

    return df


#operations = df.reset_index().groupby('COMTRS')['OPERATOR_ID'].agg(lambda x: len(set(x)))

#df['num_operations'] = 0 
#df.loc[operations.index, 'num_operations']
import sys



#%%
def fit_model(subset, 
              #organic_est, 
              area_eff_col):
    with pm.Model() as model:
        subset = subset.dropna(subset = ['ACRES_PLANTED'])
        county_idx, county_dict = make_random_effects(subset, 
                                                      area_eff_col)
        
        year_idx, year_dict = make_random_effects(subset, 'YEAR')
        #organic = subset[organic_est].values
        lodi = subset['perc_certified_lodi'].fillna(0).values
        
        #Estimate total acres of grapes based on two data sources
        acres_PUR = subset["ACRES_PLANTED"].fillna(0).values
        #f = pm.HalfCauchy('f', 3)
        acres_geodata = subset['acres_grapes'].fillna(0).values
        geodata_avail = subset['geodata_available'].values
        weight_1 = 1        
        # if geodata grape acreage is available weight it with acres_PUR
        #if not just use acres_PUR
        acres_grapes = (acres_PUR*weight_1 + acres_geodata*(1-weight_1))*geodata_avail \
            + acres_PUR*((geodata_avail - 1)*-1)
        
        #estimate organic as an ensemble of the two estimates 
        weight_2 = pm.Beta('weight_farms', alpha =2, beta =2)
        #weight_2 = .75
        organic_cell = subset['perc_certified_org'].fillna(0).values
        organic_neigh = subset['perc_certified_neigh_org'].fillna(0).values
        organic = organic_cell*weight_2 + organic_neigh * (1-weight_2) 
        
        
        
        
      
        #lbs_ai = np.log(subset[lbs_per_ac_est]+.01)
        
        
        lbs_ai = subset["LBS_AI"].values
        year_slope = pm.Normal('y_slope', mu =0, sigma =10)
        napa = subset['perc_certified_napa'].values
        #year_sig = pm.HalfCauchy('y_sig', 3)
        #year_effects = pm.Normal('y_eff', mu = 0, sigma = year_sig, 
        #                      shape = max(year_idx)+1)
        
        b_org = pm.Normal('b_org', mu = 0, sigma = 10)
        b_lodi = pm.Normal('b_lodi', mu = 0, sigma = 10)
        b_napa = pm.Normal('b_napa', mu = 0, sigma = 10)
        a = pm.Normal('a', mu = 0, sigma = 10)
        
        sigma = pm.HalfCauchy('sig', 3)
        
        county_sig = pm.HalfCauchy('c_sig', 3)
        county_effects = pm.Normal('c_eff', mu = 0, 
                                sigma = county_sig, shape = max(county_idx)+1)
        
        #est_LBS_AI = a  + year_effects[year_idx] + county_effects[county_idx]           
        
        est_LBS_AI = (a  + b_org * organic + b_lodi * lodi \
            + year_slope*year_idx + county_effects[county_idx] + napa * b_napa)\
            * acres_grapes
        
        
        likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = sigma, 
                           observed = lbs_ai)
        trace = pm.sample(
                    init = 'advi', 
                      cores = 8,
                      draws = 1500, tune = 1500, 
                      )
        
    return trace, county_dict, year_dict

#%%
def report_means_CI(data, name, t):
    return f'Estimated impact of {name} on {t}: {data.mean()}, ({np.quantile(data, .025)} - {np.quantile(data, .975)})'


def run_and_analyze(subset, area_eff_col, r_dir,
                    tracked_vars = ['b_org', 'b_lodi', 'b_napa', 
                                           #'weight_area',
                                           'weight_farms',
                                           'y_slope'
                        ]):
    
    
    with open(os.path.join(r_dir, f'summary_{area_eff_col}.txt'), 'w+') as out_file:
        trace, county_dict, year_dict = fit_model(subset, 
                                                        area_eff_col)
        #y_effs = trace.get_values('y_eff', burn =500, thin = 2)
        #year_mean = y_effs.mean(0)
        
        pm.plots.traceplot(trace, var_names = tracked_vars)
        
        
        
        plt.suptitle(f'Estimates using: {area_eff_col}')
        plt.savefig(os.path.join(r_dir, f'Traceplot_{area_eff_col}.png'), 
                    bbox_inches = 'tight')
        plt.close()
        
        #pm.plots.plot_posterior(trace, var_names = ['b_org', 'b_lodi', 'sig', 'a'])
        #plt.show()
        #print(pm.stats.summary(trace))
        ax = az.plot_forest(trace, var_names=["c_eff"])
        ax[0].set_yticklabels(list(county_dict.keys()))
        plt.savefig(os.path.join(
            r_dir, f'Forest_plot_{area_eff_col}.png'),
                    bbox_inches = 'tight')
        #plt.title(f'Estimates using: {org_est}, {area_eff_col}')
        
        plt.close()
        
        
        
        b_org = trace.get_values('b_org')
        b_lodi = trace.get_values('b_lodi')
        b_napa = trace.get_values('b_napa')
        c_effs = trace.get_values('c_eff')
        pickle.dump(trace, open(os.path.join(r_dir, f'trace_{area_eff_col}.p'),'wb'))
        print(f'Results for  {area_eff_col}', file = out_file)
        print(report_means_CI(b_org, "Organic Farming", t), file = out_file)
        print(report_means_CI(b_lodi, "LODI RULES", t), file = out_file)
        print(report_means_CI(b_napa, 'NAPAGREEN', t), file = out_file)
        
        
        
        print('Area Effects:', file = out_file)
        
        #y_slope = trace.get_values('y_slope', thin =3)
        mean_county = list(np.round(c_effs.mean(0), 3))
        low_county = list(np.round(np.quantile(c_effs, .025, axis =0),3))
        high_county = list(np.round(np.quantile(c_effs, .975, axis =0), 3))
        for county, mean, lo, hi in zip(county_dict.keys(), mean_county, 
                                low_county, high_county):
            print (f'{county}:    {mean} ({lo} - {hi}) ', file =out_file)
        
             
            #county_vals = subset.replace(county_dict)
            
            #subset['county_effect'] = subset[area_eff_col].apply(lambda x: mean_county[county_dict.get(x)])
            
            #subset['year_eff'] = subset['YEAR'].apply(lambda x: year_mean[year_dict.get(x)])
            #subset['log'] = np.log(subset[lbs_per_ac_est])
            #subset['predicted'] = a.mean() + b.mean()* subset[organic_est]  + subset['year_eff'] +subset['county_effect']
            #subset['residual'] = subset['log'] - subset['predicted']
            
            #plt.scatter(subset['predicted'], subset['residual'])
            
            #s = subset[subset['perc_certified']==0]
            #s1 = subset[subset['perc_certified']>=.01]
            #plt.scatter(s['predicted'], s['residual'])
            #plt.scatter(s1['predicted'], s1['residual'])
            #plt.show()
            #plt.scatter(s1[organic_est], s1['predicted'])
            #plt.show()
            #plt.scatter(subset[organic_est], subset['residual'])

#%%

for t in [
        'HERBICIDE', 
        'INSECTICIDE', 
        'FUNGICIDE']:

    
    #for t in ['Herbicide', "Insecticide", "Rodenticide", "Fungicide", 'Other']:   
    lbs_per_ac_ests =['LBS_PER_AC_area_from_PUR', 'LBS_PER_AC_area_from_geodata']
    organic_ests = ['perc_certified_org', 'perc_certified_neigh_org']
    area_eff_cols = ['INACNAME', 
                     #'County'
                     ]
    r_dir = os.path.join('results', t)
    if not os.path.exists(r_dir):
        os.makedirs(r_dir)

    #sys.stdout = open(os.path.join(r_dir, 'summaries.txt'), 'w+')
    for   area_eff_col in  area_eff_cols:
         
         #fit_data = fit_data[fit_data[lbs_per_ac_est]>0]
         #subset= subset[subset[lbs_per_ac_est]<20]
         subset = filter_data(data, 'PRODNO', type_df, 'TYPEPEST_CAT', t)
         subset = prep_gb(subset,
                          data).reset_index(
            ).replace({np.inf: np.nan, np.inf*-1: np.nan}
                      ).dropna(subset = ['LBS_AI', 'ACRES_PLANTED'])
                      
         
         run_and_analyze(subset, area_eff_col, r_dir)
         del subset
#%%      
    #sys.stdout.close()
for t in [
        'HERBICIDE', 
        'INSECTICIDE', 
        'FUNGICIDE']:
    for signl_code, name  in zip((2,3,4), ('DANGER', 'WARNING', 'CAUTION')):
    
        r_dir = os.path.join('results', t+'_'+name)
        if not os.path.exists(r_dir):
            os.makedirs(r_dir)
        
        sys.stdout = open(os.path.join(r_dir, 'summaries.txt'), 'w+')
        for   area_eff_col in  area_eff_cols:
            print(f'Results for  {area_eff_col}')
            subset = data[data['SIGNLWRD_IND']<=signl_code]
            subset = filter_data(data, 'PRODNO', 
                                 type_df, 'TYPEPEST_CAT', t)
            subset = prep_gb(subset,
                          data).reset_index(
            ).replace({np.inf: np.nan, np.inf*-1: np.nan}
                      ).dropna(subset = ['LBS_AI'])
                      
            
            run_and_analyze(subset, area_eff_col, r_dir)
            del subset