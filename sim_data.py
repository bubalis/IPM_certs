#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:48:52 2021

@author: bdube
"""
import pandas as pd
import numpy as np
from scipy import stats
import pymc3 as pm

import matplotlib.pyplot as plt

perc_cert = np.linspace(.075, .75, 30)- (np.random.random(30)/7)
perc_cert = np.where(perc_cert<0, 0, perc_cert)
perc_cert = {y: perc_cert[i] for i, y in enumerate(range(1990, 2020))}

sim_df = pd.concat([pd.DataFrame({'COMTRS': range(0, 500), 'LODI': [0]*400 + [1]*100, 'YEAR': y, 
                                  'Region': np.array([[i]*100 for i in range(0,5)]).flatten()},
                                 )
                     for y in range(1990, 2020)])

sim_df.reset_index(drop= True, inplace = True)

sim_df['perc_cert']= sim_df.apply(lambda row: perc_cert.get(row['YEAR']) if row['LODI'] else 0,
                                  axis = 1)
    

def sim_applications(perc_cert, num_blocks=10):
    if perc_cert == 0:
        return np.sum([non_cert_app() for i in range(num_blocks)])
    else: 
        return np.sum([app(np.random.random()<perc_cert) for i in range(num_blocks)])

def app(certified):
    if certified:
        return cert_app()
    else:
        return non_cert_app()

def non_cert_app():
    return stats.norm(4, .8).rvs()

def cert_app():
    return stats.norm(3.5, .8).rvs()

sim_df['LBS_AI'] = sim_df['perc_cert'].apply(sim_applications) + ((sim_df['YEAR']-1990)/4)

sim_df['Year_Lodi']= sim_df.apply(lambda row: f'{row["YEAR"]}_{row["LODI"]}', axis =1)

sim_df['acres_grapes'] = stats.norm(200, 40).rvs(sim_df.shape[0])
sim_df['LBS_AI'] = (sim_df['LBS_AI'] + stats.norm(0, 1).rvs(sim_df.shape[0])) * sim_df['acres_grapes']
#sim_df.to_csv('simulated_data.csv')
#%%
year_idx = sim_df['YEAR'].replace({y:i for i, y in 
                                   enumerate(sim_df['YEAR'].unique())}).values
n_years = len(sim_df.YEAR.unique())
lodi_idx = sim_df['LODI'].values 

sim_df


#model based on percent certified each year
with pm.Model() as pooled_model:
    acres = sim_df['acres_grapes'].values
    lbs_ai = sim_df['LBS_AI'].values/acres
    perc_cert = sim_df['perc_cert'].values
    coef_a = pm.Normal("coef_a", mu=0, sigma=10)
    sigma_a = pm.HalfCauchy("sigma_a", 10)
    intercept_a = pm.Normal('intercept_a', mu = 0, sigma = 10)
    
   
    a = pm.Deterministic('a', intercept_a + coef_a * year_idx + sigma_a)

    b = pm.Normal('b', mu = 0, sigma = 10)
    eps = pm.HalfCauchy("eps", 20)
    
    
    
    est_LBS_AI = a + b*perc_cert
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = eps, 
                           observed =lbs_ai)
    step = pm.Metropolis()
    trace = pm.sample(
                    #init = 'adapt_diag', 
                      cores = 8,
                      draws = 1500, tune = 1000, 
                      step = step)



#%%
year_idx = sim_df['YEAR'].replace({y:i for i, y in 
                                   enumerate(sim_df['YEAR'].unique())}).values
n_years = len(sim_df.YEAR.unique())
lodi_idx = sim_df['LODI'].values 

sim_df


#model based on percent certified each year
with pm.Model() as pooled_model:
    acres = sim_df['acres_grapes'].values
    lbs_ai = sim_df['LBS_AI'].values/acres
    perc_cert = sim_df['perc_cert'].values
    region_idx = sim_df['Region'].values
    coef_a = pm.Normal("coef_a", mu=0, sigma=10)
    
    sigma_a = pm.HalfCauchy("sigma_a", 10)
    intercept_a = pm.Normal('intercept_a', mu = 0, sigma = 10)
    
   
    a = pm.Deterministic('a', intercept_a + coef_a * year_idx + sigma_a)

    b = pm.Normal('b', mu = 0, sigma = 10)
    eps = pm.HalfCauchy("eps", 20)
    
    mu_c = pm.Normal('mu_c', mu= 0, sigma =10)
    sigma_c = pm.HalfCauchy('sigma_c', 10)
    
    c = pm.Normal('c', mu = mu_c, sigma = sigma_c, shape = 5 )
    est_LBS_AI = year_idx *c[region_idx] + b*perc_cert  
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = eps, 
                           observed =lbs_ai)
    step = pm.Metropolis()
    trace = pm.sample(
                    #init = 'adapt_diag', 
                      cores = 8,
                      draws = 1500, tune = 1000, 
                      step = step)

#%%




sim_df['year_area'] = sim_df.apply(lambda row: f'{row["YEAR"]}_{row["LODI"]}', axis =1)

#%%
sim_df.sort_values(['LODI', 'YEAR'], inplace =True)
year_area_codes = {e: i for i, e in enumerate(sim_df['year_area'].unique())}
with pm.Model() as pooled_model2:
    acres = sim_df['acres_grapes'].values
    lbs_ai = sim_df['LBS_AI'].values/acres
    year_area_idx =  sim_df['year_area'].apply(year_area_codes.get).values
    uniq_vals = sim_df['year_area'].unique().shape[0]
    
    coef_a = pm.Normal("coef_a", mu=0, sigma=10)
    sigma_a = pm.HalfCauchy("sigma_a", 10)
    intercept_a = pm.Normal('intercept_a', mu = 0, sigma = 10)
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=10)
    sigma_b = pm.HalfCauchy("sigma_b", 10)
   
    a = pm.Deterministic('a', intercept_a + coef_a * year_idx + sigma_a)
    
    # Intercept for each year, distributed around group mean mu_a
    b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=uniq_vals)
    #b = pm.Normal('b', mu = 0, sigma = 10)
    eps = pm.HalfCauchy("eps", 20)
    
    
    
    est_LBS_AI = b[year_area_idx]
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = eps, 
                           observed =lbs_ai)
    step = pm.Metropolis()
    trace = pm.sample(
                    #init = 'adapt_diag', 
                      cores = 8,
                      draws = 3500, tune = 1500, step = step)
    


x = trace.get_values('b')
lodi_x = x[:, 30:60]
non_lodi_x = x[:, 0:30]
#%%
plt.plot(non_lodi_x.mean(axis = 0), color = 'b')
plt.fill_between(list(range(0, 30)), 
                 np.quantile(non_lodi_x, .05, axis = 0), 
                 np.quantile(non_lodi_x, .95, axis = 0), color = 'b', 
                 alpha =.2)


plt.plot(lodi_x.mean(axis = 0), color = 'r')
plt.fill_between(list(range(0, 30)), 
                 np.quantile(lodi_x, .05, axis = 0), 
                 np.quantile(lodi_x, .95, axis = 0), color = 'r',
                 alpha =.2)
#%%
gb_is = sim_df.groupby(['LODI', 'YEAR'])[['LBS_AI', 'acres_grapes']].sum().reset_index()
year_idx = gb_is['YEAR'].replace({y:i for i, y in 
                                   enumerate(sim_df['YEAR'].unique())}).values
gb_is['year_area'] = gb_is.apply(lambda row: f'{int(row["YEAR"])}_{int(row["LODI"])}', axis =1)

#gb_is['year_area'] = gb_is.apply(lambda row: f'{row["YEAR"]}_{row["LODI"]}', axis =1)
with pm.Model() as model3:
    acres = gb_is['acres_grapes'].values
    lbs_ai = gb_is['LBS_AI'].values/acres
    is_lodi = gb_is['LODI'].values
    year_area_idx =  gb_is['year_area'].apply(year_area_codes.get).values
    coef_a = pm.Normal("coef_a", mu=0, sigma=10)
    sigma_a = pm.HalfCauchy("sigma_a", 10)
    intercept_a = pm.Normal('intercept_a', mu = 0, sigma = 10)
    
    
    a = pm.Deterministic('a', intercept_a + coef_a * year_idx + sigma_a)
    
    # Intercept for each year, distributed around group mean mu_a
    #b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=uniq_vals)
    b = pm.Normal('b', mu = 0, sigma = 10)
    eps = pm.HalfCauchy("eps", 20)
    
    
    
    est_LBS_AI = a + b * is_lodi
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = eps, 
                           observed =lbs_ai)
    step = pm.Metropolis()
    trace = pm.sample(
                    #init = 'adapt_diag', 
                      cores = 8,
                      draws = 1500, tune = 1500, step = step)
    
            
