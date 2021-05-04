#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:14:07 2021

@author: bdube
"""


import pandas as pd
import os
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import scipy as sp
import numpy as np
import pickle



    

#%%


def make_random_effects(df, col):
    '''Pass a dataframe and a column. 
    Return a vector of ordinal values corresponding with the sorted unique values
    of the column, and a dictionary with those codes.
    
    df:
        a
    0   Sam
    1   Sam 
    2   Ben 
    3   Meg
    4   Meg
    5   Sam
    6   Ben
    
    make_random_effects(df, 'a')
    returns  a series:
        
    0   2
    1   2
    2   0
    3   1
    4   1
    5   2
    6   0
    
    and a dictionary: {0: 'Ben', 1: 'Meg', 2: 'Sam'}
    '''
    
    dic = {e: i for i, e in enumerate(np.sort(df[col].unique()))}
    return df[col].replace(dic).values, dic



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
    




def prep_gb(sub_data, full_data, logit = False):
    '''Regroup pesticide data aggregated based on COMTRS and 
    make necessary calcuated columns.'''
    
    gb_columns = ['COMTRS', 'YEAR']
    
    data_cols_1 = ['LBS_AI']
    data_cols_2 = ['NAME',  'num_operations', 'num_operations_neigh',  'acres_grapes', 
                   'INACNAME',
                   'N_organic',
                        'N_organic_neigh',  
                       'perc_certified_lodi', 'perc_certified_napa', 'N_sip_cert',
                       'N_sip_neigh'
                       ]
    
    fillna_cols = ['N_organic',
                        'N_organic_neigh',  
                       'perc_certified_lodi', 'perc_certified_napa', 'N_sip_cert',
                       'N_sip_neigh', 'LBS_AI']
    
    if logit:
        gb_columns.append('GROWER_CODE')
        data_cols_1.append('AREA_PLANTED')
        
    
    df = sub_data.groupby(gb_columns
                       )[data_cols_1].sum()
    
    
    #collect by-tract production data, which should include all tracts.
    #not just tracts that had entries for chemicals of interest
    gb = full_data.groupby(gb_columns) 
    df = df.merge(gb[data_cols_2].agg('first'),
                  how= 'right', left_index= True, right_index = True)
    
    if not logit:
        df = df.merge(gb[['AREA_PLANTED']].agg(lambda x: x.unique().sum()),
                  how = 'right', left_index= True, right_index = True)
    
    df[ fillna_cols] = df[ fillna_cols].fillna(0)
    
    
    df.rename(columns = {'NAME': 'County', 'AREA_PLANTED': 'ACRES_PLANTED'}, inplace =True)
    
    df['LBS_PER_AC'] =  df['LBS_AI']/ df['ACRES_PLANTED']
    
    
    filter_1 = lambda x: x if x<=1 else 1 #constrain 1 as the highest value. 
    
    #flag for whether raster data of grape acreage is present
    df['geodata_available'] = (df['acres_grapes'].isna() == False).astype(int)
    
    #calculate ratio of organic operations to total operations
    df['perc_certified_org'] = (df['N_organic'] / df['num_operations']
                                ).apply(filter_1)
    df['perc_certified_neigh_org'] = (df['N_organic_neigh'] / df['num_operations_neigh']
                                      ).apply(filter_1)

    df['perc_certified_sip'] = (df['N_sip_cert'] / df['num_operations']
                                ).apply(filter_1)
    df['perc_certified_neigh_sip'] = (df['N_sip_neigh'] / df['num_operations_neigh']
                                      ).apply(filter_1)
    
    
    return df.reset_index(
        ).replace({np.inf: np.nan, np.inf*-1: np.nan}
                  ).dropna(subset = ['LBS_AI', 'ACRES_PLANTED'])


#operations = df.reset_index().groupby('COMTRS')['OPERATOR_ID'].agg(lambda x: len(set(x)))

#df['num_operations'] = 0 
#df.loc[operations.index, 'num_operations']



#%%

def make_year_spline(subset, k = 5):
    year_sum = subset.groupby('YEAR').sum()[['ACRES_PLANTED', "LBS_AI"]].sort_index()
    year_sum['LBS_per_ac'] = year_sum['LBS_AI']/year_sum['ACRES_PLANTED']

    year_sum.reset_index(inplace = True)
    return sp.interpolate.UnivariateSpline(year_sum.index, year_sum['LBS_per_ac'],
                                           k = k)
    


def fit_model(subset, 
              #organic_est, 
              area_eff_col, log = True):
    '''Linear model for predcicting lbs_ai of the subsetted data.
    Subset: data to fit to
    area_eff_col: area-grouping for area random-effects. 
    log: Whether to log-transform the data.'''
    
    
    year_idx, year_dict = make_random_effects(subset, 'YEAR')
    
    #attempt at adding a spline for time-series
    
    
    with pm.Model() as model:
        #subset = subset.dropna(subset = ['ACRES_PLANTED'])
        county_idx, county_dict = make_random_effects(subset, 
                                                      area_eff_col)
        
        
        #organic = subset[organic_est].values
        
        
        #Estimate total acres of grapes based on two data sources
        #acres_PUR = subset["ACRES_PLANTED"].fillna(0).values
        #f = pm.HalfCauchy('f', 3)
        #acres_geodata = subset['acres_grapes'].fillna(0).values
        #geodata_avail = subset['geodata_available'].values
        #weight_1 = 1        
        # if geodata grape acreage is available weight it with acres_PUR
        #if not just use acres_PUR
        #acres_grapes = (acres_PUR*weight_1 + acres_geodata*(1-weight_1))*geodata_avail \
        #    + acres_PUR*((geodata_avail - 1)*-1)
        
        #estimate organic as an ensemble of the two estimates 
        weight_2 = pm.Beta('weight_farms', alpha =3, beta =3)
        
        #weight_2 = .75
        organic_cell = subset['perc_certified_org'].fillna(0).values
        organic_neigh = subset['perc_certified_neigh_org'].fillna(0).values
        sip_cell =  subset['perc_certified_sip'].fillna(0).values
        sip_neigh = subset['perc_certified_neigh_sip'].fillna(0).values
        
        sip = sip_cell * weight_2 + sip_neigh * (1-weight_2)
        organic = organic_cell*weight_2 + organic_neigh * (1-weight_2) 
        
        lbs_per_ac = subset['LBS_PER_AC'].values
        if log:
            lbs_per_ac_min = subset[subset['LBS_PER_AC']>0]['LBS_PER_AC'].min()
            print(lbs_per_ac_min)
            lbs_per_ac = np.log(lbs_per_ac_min/2 +lbs_per_ac)
    
        #year_slope = pm.Normal('y_slope', mu =0, sigma =10)
        napa = subset['perc_certified_napa'].fillna(0).values
        lodi = subset['perc_certified_lodi'].fillna(0).values
        
        
        year_sig = pm.HalfCauchy('y_sig', 3)
        y_eff = pm.Normal('y_eff', mu = 0, sigma = year_sig, 
                              shape = max(year_idx)+1)
        
        b_org = pm.Normal('b_org', mu = 0, sigma = 10)
        b_lodi = pm.Normal('b_lodi', mu = 0, sigma = 10)
        b_napa = pm.Normal('b_napa', mu = 0, sigma = 10)
        b_sip = pm.Normal('b_sip', mu = 0, sigma = 10)
        
        a = pm.Normal('a', mu = 0, sigma = 10)
        sigma = pm.HalfCauchy('sig', 3)
        
        county_sig = pm.HalfCauchy('c_sig', 3)
        county_effects = pm.Normal('c_eff', mu = 0, 
                                sigma = county_sig, shape = max(county_idx)+1)
        
        est_LBS_AI = a  + b_org * organic + b_lodi * lodi + b_sip * sip \
            + y_eff[year_idx] + county_effects[county_idx] + napa * b_napa
          
        likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = sigma, 
                           observed = lbs_per_ac)
        
        trace = pm.sample(
                    init = 'advi', 
                      cores = 8,
                      draws = 1500, tune = 1000, 
                      )
        
    return trace, county_dict, year_dict


def fit_dummy_model(subset, area_eff_col, log = True):
    '''Dummy model for testing basic parameterization.
    Tries to predict soley using the estimates of organic prevalence. 
    Subset: data to fit to
    area_eff_col: area-grouping for area random-effects. 
    log: Whether to log-transform the data.
    '''
    
    
    year_idx, year_dict = make_random_effects(subset, 'YEAR')
    
    #attempt at adding a spline for time-series
    
    #spl = make_year_spline(subset)
    
    
    with pm.Model() as model:
        #subset = subset.dropna(subset = ['ACRES_PLANTED'])
        county_idx, county_dict = make_random_effects(subset, 
                                                      area_eff_col)
        
        
        #organic = subset[organic_est].values
        
        #Estimate total acres of grapes based on two data sources
        #acres_PUR = subset["ACRES_PLANTED"].fillna(0).values
        #f = pm.HalfCauchy('f', 3)
        #acres_geodata = subset['acres_grapes'].fillna(0).values
        #geodata_avail = subset['geodata_available'].values
        lbs_per_ac = subset['LBS_PER_AC'].values
       
        
        if log:
            lbs_per_ac_min = subset[subset['LBS_PER_AC']>0]['LBS_PER_AC'].min()
            print(lbs_per_ac_min)
            lbs_per_ac = np.log(lbs_per_ac_min/2 +lbs_per_ac)
        
        weight_1 = 1        
        # if geodata grape acreage is available weight it with acres_PUR
        #if not just use acres_PUR
        #acres_grapes = (acres_PUR*weight_1 + acres_geodata*(1-weight_1))*geodata_avail \
        #    + acres_PUR*((geodata_avail - 1)*-1)
        #acres_grapes = acres_PUR
        
        #estimate organic as an ensemble of the two estimates 
        weight_2 = pm.Beta('weight_farms', alpha =2, beta =2)
        #weight_2 = .75
        organic_cell = subset['perc_certified_org'].fillna(0).values
        organic_neigh = subset['perc_certified_neigh_org'].fillna(0).values
        organic = organic_cell*weight_2 + organic_neigh * (1-weight_2) 
        
        #lbs_ai = subset["LBS_AI"].values/acres_PUR
        #year_slope = pm.Normal('y_slope', mu =0, sigma =10)
        
        year_sig = pm.HalfCauchy('y_sig', 3)
        year_effects = pm.Normal('y_eff', mu = 0, sigma = year_sig, 
                              shape = max(year_idx)+1)
        
        b_org = pm.Normal('b_org', mu = 0, sigma = 3)
        a = pm.Normal('a', mu = 0, sigma = 10)
        
        
        
        sigma = pm.HalfCauchy('sig', 3)
        
        county_sig = pm.HalfCauchy('c_sig', 3)
        county_effects = pm.Normal('c_eff', mu = 0, 
                               sigma = county_sig, shape = max(county_idx)+1)
        
        #est_LBS_AI = a  + year_effects[year_idx] + county_effects[county_idx]           
        
        est_LBS_AI = a  + b_org * organic + year_effects[year_idx]  + county_effects[county_idx]
            
        
        likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = sigma, 
                           observed = lbs_per_ac)
        
        trace = pm.sample(
                    init = 'advi', 
                      cores = 4,
                      draws = 500, tune = 500, 
                      )
        
    return trace


def fit_logit_model(subset, area_eff_col, dummy = False):
    '''A logit model for predicting whether a class of product is NOT USED
    within a given plot-year. '''
    
    year_idx, year_dict = make_random_effects(subset, 'YEAR')
    county_idx, county_dict = make_random_effects(subset, 
                                                      area_eff_col)
    #attempt at adding a spline for time-series
    
    #spl = make_year_spline(subset)
    with pm.Model() as model:
        #subset = subset.dropna(subset = ['ACRES_PLANTED'])
        
        weight_2 = pm.Beta('weight_farms', alpha =2, beta =2)
        if not dummy:
            area = subset['acres_grapes'].values
            
            napa = subset['perc_certified_napa'].fillna(0).values
            lodi = subset['perc_certified_lodi'].fillna(0).values
            sip_cell =  subset['perc_certified_sip'].fillna(0).values
            sip_neigh = subset['perc_certified_neigh_sip'].fillna(0).values
        
            sip = sip_cell * weight_2 + sip_neigh * (1-weight_2) 
            
            b_lodi = pm.Normal('b_lodi', mu = 0, sigma = 10)
            b_napa = pm.Normal('b_napa', mu = 0, sigma = 10)
            b_area = pm.Normal('b_area', mu = 0, sigma = 10)
            b_sip = pm.Normal('b_sip', mu= 0, sigma = 10)
            
            
        #estimate organic as an ensemble of the two estimates 
        
        #weight_2 = .75
        organic_cell = subset['perc_certified_org'].fillna(0).values
        organic_neigh = subset['perc_certified_neigh_org'].fillna(0).values
        organic = organic_cell*weight_2 + organic_neigh * (1-weight_2) 
        
        lbs_ai = subset["LBS_AI"]==0
        #year_slope = pm.Normal('y_slope', mu =0, sigma =10)
        
        year_sig = pm.HalfCauchy('y_sig', 3)
        year_effects = pm.Normal('y_eff', mu = 0, sigma = year_sig, 
                              shape = max(year_idx)+1)
        
        b_org = pm.Normal('b_org', mu = 0, sigma = 10)
        
        
        a = pm.Normal('a', mu = 0, sigma = 10)
        
        
        
        #sigma = pm.HalfCauchy('sig', 3)
        
        county_sig = pm.HalfCauchy('c_sig', 3)
        county_effects = pm.Normal('c_eff', mu = 0, 
                               sigma = county_sig, shape = max(county_idx)+1)
        
        #est_LBS_AI = a  + year_effects[year_idx] + county_effects[county_idx]           
        if dummy:
            p = pm.Deterministic('p',
                             pm.math.sigmoid(
    b_org*organic + a + year_effects[year_idx]+county_effects[county_idx]
                                                           ))
        else:
            p = pm.Deterministic('p',
                             pm.math.sigmoid(
    b_org*organic + b_lodi*lodi+ b_napa*napa + a + year_effects[year_idx] +sip *b_sip
                                                           ))
            
        
        likelihood = pm.Bernoulli('likelihood',  
                            p, 
                           observed = lbs_ai)
        
        if dummy:
            draws = 500
            tune = 500
            cores = 6
        
        else:
            draws = 1000
            tune = 1500
            cores = 8
        
        trace = pm.sample(
                    init = 'advi', 
                      cores = cores,
                      draws = draws, tune = tune, 
                      )
        
    return trace, county_dict,  year_dict
    
#%%
def report_means_CI(data, name, t):
    return f'Estimated impact of {name} on {t}: {data.mean()}, ({np.quantile(data, .025)} - {np.quantile(data, .975)})'


def run_and_analyze(subset, area_eff_col, r_dir, logit =False,
                    tracked_vars = ['b_org', 'b_lodi', 'b_napa', 'b_sip',
                                           #'weight_area',
                                            'weight_farms',
                                           #'y_slope'
                        ]):
    
    
    with open(os.path.join(r_dir, f'summary_{area_eff_col}.txt'), 'w+') as out_file:
        print('Total lbs of AI applied:', subset['LBS_AI'].sum(), file = out_file)
        print('Total COMTRS with applications:', (subset['LBS_AI']>0).sum(), out_file)
        if not logit:
            trace, county_dict, year_dict = fit_model(subset, 
                                                        area_eff_col)
        else:
            trace, county_dict, year_dict = fit_logit_model(subset, area_eff_col)
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
        b_sip = trace.get_values('b_sip')
        c_effs = trace.get_values('c_eff')
        pickle.dump(trace, open(os.path.join(r_dir, f'trace_{area_eff_col}.p'),'wb'))
        print(f'Results for  {area_eff_col}', file = out_file)
        print(report_means_CI(b_org, "Organic Farming",use_type), file = out_file)
        print(report_means_CI(b_lodi, "LODI RULES",use_type), file = out_file)
        print(report_means_CI(b_napa, 'NAPAGREEN',use_type), file = out_file)
        print(report_means_CI(b_sip, 'SIP',use_type), file = out_file)
        
        
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


if __name__ == '__main__':
    os.chdir('spatial')
    data = pd.read_csv(os.path.join('intermed_data', 'ready_for_analysis.csv'))
    type_df = pd.read_csv(os.path.join('intermed_data', 'pesticide_types.csv'))
    '''
    ### Checks using just organic as a predictor
    ### Testing with and without log transformation
    r_dir = os.path.join('results', 'organic_check')
    if not os.path.exists(r_dir):
        os.makedirs(r_dir)
        
    subset_1 = data[data['GEN_PEST_IND']=='M']
    subset_2 = filter_data(data, 'PRODNO', type_df, 'TYPEPEST_CAT', 'HERBICIDE')
    for subset, subset_name in zip( 
            (subset_1, subset_2),
            ('organic_microbe', 'organic_herbicide')):
        
        subset = prep_gb(subset, data)
        for _bool in (True, False):
            log_flag = 'log'*_bool
            
        
        
        
            trace = fit_dummy_model(subset, 'INACNAME', _bool)     
            pm.plots.traceplot(trace, var_names = ['b_org'])
            plt.savefig(os.path.join(r_dir, f'trace_{subset_name}_{log_flag}.png'), 
                        bbox_inches = 'tight')
            plt.close()
        
        trace, _ , _ = fit_logit_model(subset, 'INACNAME', dummy = True)
        pm.plots.traceplot(trace, var_names = ['b_org'])
        plt.savefig(os.path.join(r_dir, f'trace_{subset_name}_logit.png'), 
                        bbox_inches = 'tight')
        plt.close()
        '''
    #%%
    
    
    
    
              
    
    area_eff_cols = ['INACNAME', 
                         #'County'
                         ]
    
    ###total chemicals applied by category
    
    '''
    for use_type in [
            'HERBICIDE', 
            'INSECTICIDE', 
            'FUNGICIDE']:
        #lbs_per_ac_ests =['LBS_PER_AC_area_from_PUR', 'LBS_PER_AC_area_from_geodata']
        #organic_ests = ['perc_certified_org', 'perc_certified_neigh_org']
        
        r_dir = os.path.join('results', use_type)
        if not os.path.exists(r_dir):
            os.makedirs(r_dir)
    
        #sys.stdout = open(os.path.join(r_dir, 'summaries.txt'), 'w+')
        for   area_eff_col in  area_eff_cols:
             
            #fit_data = fit_data[fit_data[lbs_per_ac_est]>0]
            #subset= subset[subset[lbs_per_ac_est]<20]
            subset = filter_data(data, 
                                    'PRODNO', 
                                    type_df, 'TYPEPEST_CAT',use_type)
            if use_type!= 'FUNGICIDE':
                subset = subset[subset['PRODUCT_NAME'].str.contains('SULFUR')==False]
            
            subset = prep_gb(subset,
              data)
              
             
            run_and_analyze(subset, area_eff_col, r_dir)
            del subset
            '''
        
        #sys.stdout.close()
    for use_type in [
            'INSECTICIDE', 
            'HERBICIDE', 
            
            'FUNGICIDE']:
        '''
        for signl_code, name  in zip((3,4), ('WARNING', 'CAUTION')):
        
            r_dir = os.path.join('results', use_type +'_'+name)
            if not os.path.exists(r_dir):
                os.makedirs(r_dir)
            
            #sys.stdout = open(os.path.join(r_dir, 'summaries.txt'), 'w+')
            for   area_eff_col in  area_eff_cols:
                
                subset = filter_data(data[data['SIGNLWRD_IND']<=signl_code], 
                                     'PRODNO', 
                                     type_df, 'TYPEPEST_CAT',use_type)
                if use_type!= 'FUNGICIDE':
                    subset = subset[subset['PRODUCT_NAME'].str.contains('SULFUR')==False]
                
                subset = prep_gb(subset,
                              data, logit = False)
                
                run_and_analyze(subset, area_eff_col, r_dir)
                del subset
                '''
        signl_code = 2
        name = 'DANGER'
        r_dir = os.path.join('results', use_type +'_'+name)
        if not os.path.exists(r_dir):
            os.makedirs(r_dir)
        for   area_eff_col in  area_eff_cols:
            subset = filter_data(data[data['SIGNLWRD_IND']<=signl_code], 
                                 'PRODNO', 
                                 type_df, 'TYPEPEST_CAT',use_type)
            del type_df
            if use_type!= 'FUNGICIDE':
                subset = subset[subset['PRODUCT_NAME'].str.contains('SULFUR')==False]
            
            subset = prep_gb(subset,
                          data, logit = False)
                
            run_and_analyze(subset, area_eff_col, r_dir, logit = True)
            del subset
            type_df = pd.read_csv(os.path.join('intermed_data', 'pesticide_types.csv'))
