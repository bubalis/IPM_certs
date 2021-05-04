#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:21:35 2021

@author: bdube
"""
import pandas as pd
import geopandas as gpd
import os

import matplotlib.pyplot as plt
from functools import partial



def plot_summary(subset):
    year_sum = subset.groupby('YEAR').sum()[['ACRES_PLANTED', "LBS_AI"]]
    year_sum['LBS_per_ac'] = year_sum['LBS_AI']/year_sum['ACRES_PLANTED']
    year_sum['LBS_per_ac'].plot()
    plt.show()


def try_merge(left, right, chunksize=1000, **kwargs):
    '''Merge two dataframes together, using the standard pd.merge function.
    If this results in a memory error, save the right-side dataframe 
    to a scratch csv and merge in chunks.
    functions exactly as pd.merge
    ''' 
    
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

def check_rec_bad(row):
    '''Returns True if a record is likely to be bad.
    False otherwise.
    Guidelines here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.168.6558&rep=rep1&type=pdf
    for determining if records are bad.'''
    
    
    try:
        return any([(row['LBS_PER_AC'] > row['median_rate']*50),
            ((row['FUMIGANT_SW']!='X') & (row['LBS_PER_AC']>200)),
            ((row['FUMIGANT_SW'] == 'X') & (row['LBS_PER_AC']>1000) )])
    except:
        print(row)
        raise
    
def fix_bad_recs(apps):
    '''Find bad records in the applications dataframe.
    Replace these with the median application rate for those chemicals.
    '''
    
    medians = apps.groupby('CHEMNAME')['LBS_PER_AC'].median().to_dict()
    #with mp.Pool(4) as pool:
    #    apps['median_rate'] = pool.map(medians.get, apps['CHEMNAME'])
    apps['median_rate'] = apps['CHEMNAME'].apply(medians.get)
    indicies = apps.apply(check_rec_bad, axis = 1)
    #indicies = parallelize_on_rows(apps, check_rec_bad, 4)
    
    apps.loc[indicies, "LBS_AI"] = apps.loc[indicies, 'median_rate'] * apps.loc[indicies, 'AREA_TREATED']
    apps['LBS_PER_AC'] = apps['LBS_AI']/apps['AREA_TREATED']
    return apps, indicies





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



def collect_neighbors(geometry, gdf, sindex = None, id_col = "COMTRS"):
    '''Get all the neighbors of a geometry from a gdf.
    If id_col is passed, return a list of values for id_col.
    if id_col is NOne, return the subsetted dataframe of neighbors. 
    
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



def agg_from_neighbors(indicies, gdf, agg_col, matcher_col= 'COMTRS'):
    '''Function for aggregating from listed neighbors.
    args: indicies- a list of values, either index values or values for matcher_col
    agg_col: column to aggregate from.
    matcher_col: column to identify neighbors from. If None, use index.
    '''
    
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
        #print(subset_val)
        subset = gdf[gdf[subset_col] == subset_val]
        func = partial(agg_from_neighbors, gdf = subset, 
                       agg_col = agg_col, matcher_col = matcher_col)
        
        out.append(subset[index_col].apply(func))
        # gdf.loc[subset.index, new_col_name] = subset[new_col_name]
        
    return pd.concat(out)



os.chdir('spatial')
#%%
preped_data_path = os.path.join('intermed_data', 'ready_for_analysis.csv')


if __name__ == '__main__': 
    comtrs = gpd.read_file(os.path.join('intermed_data', 'ava_comtrs'))
    comtrs['neighbors'] = comtrs['neighbors'].apply(lambda x: x.split(','))
    
    
    #assemble organic data
    org_vineyards = gpd.read_file(os.path.join('intermed_data', 
                                               'organic_vineyards_shp'))
    org_vineyards['Year'] = org_vineyards['Effective'].apply(lambda x: 
                                                             int(x.split('-')[-1]))
    
    #remove multiple entries for the same address
    org_vineyards.drop_duplicates(subset = ['search_add'], inplace = True)
    org_vineyards.to_crs(comtrs.crs, inplace = True)
    org_vineyards = org_vineyards[org_vineyards.geometry.is_valid]
    org_cert_by_year = get_cert_years(comtrs, org_vineyards, 
                                  'Year', 
                                  range(2003, 2020))
        
    
    #assemble SIP certification Data
    sip_vineyards = gpd.read_file(os.path.join('intermed_data', 'sip_farms'))
    sip_cert_by_year = get_cert_years(comtrs, 
                                      sip_vineyards, 
                                      'Year Certi', range(2003, 2020),
                                      out_name = 'N_sip_cert')
    #add in neighbor data 
    org_cert_by_year = org_cert_by_year.merge(comtrs[['COMTRS', "neighbors"]], on = 'COMTRS')
    org_cert_by_year['N_organic_neigh'] = vector_neigh_match(org_cert_by_year, 'N_organic')
    sip_cert_by_year = sip_cert_by_year.merge(comtrs[['COMTRS', "neighbors"]], on = 'COMTRS')
    sip_cert_by_year['N_sip_neigh'] = vector_neigh_match(sip_cert_by_year, 'N_sip_cert')
    #%%
    
    
    
    del org_vineyards
    
     
    acres_grapes = pd.read_csv(os.path.join('intermed_data', 'california_grape_acres.csv')
                               ).drop(columns = 'Unnamed: 0')
    acres_grapes = acres_grapes.melt(id_vars = 'COMTRS', 
                                     value_name ='acres_grapes', 
                                     var_name = 'YEAR')
    
    acres_grapes['YEAR'] = acres_grapes['YEAR'].apply(lambda x: int(
                                    x.replace('acres_grapes_', '')))
    
    
    

    
    #add a column listing the neighbors of each COMTRS
    #comtrs['neighbors'] = comtrs['geometry'].apply(partial(collect_neighbors, 
    #                                    gdf = comtrs, sindex = comtrs.sindex))
    
    
    #prep the applications data
    apps = pd.read_csv(os.path.join('source_data', 'PIP_recs.txt'), sep = '\t')
    
    apps = apps[['COMTRS', 'GROWER_ID', 
                 'AREA_PLANTED', 'YEAR', 
                 'LBS_AI', 'PRODNO', 'PRODUCT_NAME', 'UNIT_TREATED',
                 'UNIT_PLANTED', 'AREA_TREATED', 'CHEMNAME'
                 ]]
   
    
    #correct for unit_planted not being acres.
    apps.loc[ apps['UNIT_PLANTED']=='S', 'AREA_PLANTED' ] = apps.loc[
        apps['UNIT_PLANTED']=='S',  'AREA_PLANTED' ] /43560
    
    
    apps['GROWER_CODE'] = apps['GROWER_ID'].apply(
        lambda x: x[-7:] if type(x)==str else '') 
    
    
    
    product_df = pd.read_csv(os.path.join('intermed_data', 
                                         'pesticide_products.csv'))
    
    #merge in pesticide product data
    apps = apps.merge(product_df[['PRODNO', 'SIGNLWRD_IND', 'FUMIGANT_SW', 
                                  'GEN_PEST_IND']], 
                      on = 'PRODNO', how = 'left')
    
    #set lbs per acre and fix bad recs
    apps['LBS_PER_AC'] = apps['LBS_AI']/apps['AREA_TREATED']
    apps, fixed_indicies = fix_bad_recs(apps)
    apps.drop(columns = ['AREA_TREATED', "LBS_PER_AC"], inplace = True)
    
    #set all Null values to No signalword (safest level)
    apps['SIGNLWRD_IND'] = apps['SIGNLWRD_IND'].fillna(5)
    
    del product_df
    
    apps = apps[apps['YEAR']>1990]
    
    
    
    
   
    cols_to_keep = ['COMTRS',
                    'NAME',  
                    'neighbors', 
                     'INACNAME', 
                     ]
    
    comtrs = comtrs[cols_to_keep]
    
    data = try_merge(comtrs, apps, chunksize=1000,  on = 'COMTRS', 
                     how = 'inner')
    
    data = data.merge(acres_grapes, on = ['COMTRS', 'YEAR'], how = 'left')
    
    #data = data[(data['acres_grapes']>20) | (data['acres_grapes'].isna())][cols_to_keep]
    
    #data = data[cols_to_keep]
    data =  data.merge(org_cert_by_year, on = ['YEAR', 'COMTRS'], how = 'left') 
    data = data.merge(sip_cert_by_year, on = ['YEAR', 'COMTRS'], how = 'left' )
    
    del apps
    
    def len_uniq_valid_str(li):
        return len(set([i for i in li if type(i)==str and i]))
    
    data['GROWER_CODE'] = data['GROWER_CODE'].fillna('')
    #add indicator for number of distinct operations for each COMTRS each YEAR
    operations = data.groupby(['COMTRS', 'YEAR'])[['GROWER_CODE']].agg(len_uniq_valid_str
                            ).rename(columns={'GROWER_CODE': 'num_operations'}).reset_index()
    
    operations = operations.merge(
        comtrs[['COMTRS', "neighbors"]], on = 'COMTRS', how = 'right')
    
    operations['num_operations'] = operations['num_operations'].fillna(0)
    
    operations['num_operations_neigh'] = vector_neigh_match(
                                operations, 'num_operations').fillna(0)
    
    
    #add data for certified operations
    data = data.merge(operations, how = 'inner', on = ['COMTRS', 'YEAR'])
    
    data['N_organic'] = data['N_organic'].fillna(0)
    data['perc_certified_lodi'] = 0
    data['N_sip_cert'] = data['N_sip_cert'].fillna(0)
    
    #%%
    #Add in LODI certifications
    lodi_comtrs = gpd.read_file(os.path.join('intermed_data', 'lodi_comtrs'))
    lodi_cert_acres = pd.read_csv(os.path.join('source_data', 'lodi_acres.csv'))
    lodi_grape_acres = pd.read_csv(os.path.join('intermed_data', 'lodi_total_grape_acres.csv'))
    lodi_cert_by_year = lodi_cert_acres.merge(lodi_grape_acres, on = 'Year')
    
    lodi_cert_by_year['perc_certified'] = lodi_cert_by_year['lodi_acres']/lodi_cert_by_year['acres_grapes']
    
    lodi_cert_dict = dict(zip(lodi_cert_by_year['Year'].tolist(), lodi_cert_by_year['perc_certified'].tolist()))
    
    lodi_cert_perc = data.loc[
        data['COMTRS'].isin(lodi_comtrs['COMTRS'].tolist()), 'YEAR'].apply(
                                        lambda x: lodi_cert_dict.get(x, 0))
    
    data.loc[
        data['COMTRS'].isin(lodi_comtrs['COMTRS'].tolist()), 'perc_certified_lodi'] = lodi_cert_perc 
    
    
    #Napa Data:
    napa_data = pd.read_csv(os.path.join('intermed_data', 'synthetic_napa_data.csv'))
    napa_dict = dict(zip(napa_data['Year'], napa_data['perc_napagreen']))
    data['perc_certified_napa'] = 0
    napa_cert_perc = data.loc[
        data['NAME'] == 'Napa', 'YEAR'].apply(lambda x: napa_dict.get(x, 0))
    data.loc[data['NAME'] == 'Napa', 'perc_certified_napa'] = napa_cert_perc
    
    
    
    
    
    
        
    
    
    
    
    
    #data['N_organic_neigh'] = data.apply(lambda x: org_agg_func(x['neighbors'], x['YEAR']), axis =1)
    
    #ops_agg_func =  partial(agg_from_neighbors, gdf = data, agg_col = 'num_operations')
    #data['num_operations_neigh'] = ops_agg_func(data['neighbors'], data['YEAR'])
    
    #data['LBS_PER_AC'] 
    #data['LBS_PER_AC'] = data['LBS_AI']/data['acres_grapes']
    data['N_organic_neigh'].fillna(0, inplace = True)
    data['num_operations_neigh'].fillna(0, inplace = True)
    data['N_sip_neigh'].fillna(0, inplace = True)
    #data = data.merge(type_df, on = 'PRODNO')
    
    
   
    del lodi_comtrs
    del comtrs
    del acres_grapes
    del org_cert_by_year
    del sip_cert_by_year
    del operations
    
    data.drop(columns = ['neighbors_x', 'neighbors_y',
                         ], 
              inplace = True
              )
    data.to_csv(preped_data_path)