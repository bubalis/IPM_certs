#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:08:37 2021

@author: bdube
"""
import pandas as pd
import numpy as np
from scipy import stats
import pymc3 as pm

import matplotlib.pyplot as plt

#%%
sim_df = pd.concat([pd.DataFrame({'COMTRS': range(0, 500), 'is_in': [0]*400 + [1]*100, 'YEAR': y, 
                                  'Region': np.array([[i]*100 for i in range(0,5)]).flatten(),
                                 'Start_year' : [9999]*400 + [2008]*20+ [2009] *20 + [2010]*20+ [2011] *20 + [2012]*20
                                 })
                     for y in range(1990, 2020)])

#%%

plot_effects = stats.norm(1, .2).rvs(500)

plot_effects = {i: e for i, e in enumerate(plot_effects)}

year_effects = np.linspace(2, -2, 30)

year_effects = {k:v for k, v in zip(range(1990, 2020), year_effects)}

sim_df['is_certified'] = sim_df['Start_year'] <= sim_df['YEAR']
sim_df['plot_effects'] = sim_df['COMTRS'].apply(plot_effects.get)
sim_df['year_effects'] = sim_df['YEAR'].apply(year_effects.get)

#%%
def sim_applications(is_certified, num_blocks=10):
    if is_certified:
        return np.mean([cert_app() for i in range(num_blocks)])
    else:
        return np.mean([non_cert_app() for i in range(num_blocks)])
    
def cert_app():
    return stats.norm(2.5, .5).rvs()

def non_cert_app():
    return stats.norm(4, 1).rvs()


sim_df['LBS_AI'] = sim_df['is_certified'].apply(sim_applications) + sim_df['plot_effects'] + sim_df['year_effects']

sim_df['treated'] = sim_df['Start_year']<9999
#%%
year_dict = {y: i for i, y in enumerate(range(1990, 2020))}
with pm.Model() as model:
    treatment = sim_df['treated'].values
    year_idx = sim_df['YEAR'].apply(year_dict.get).values
    is_certified = sim_df['is_certified'].values
    lbs_ai = sim_df['LBS_AI'].values
    plot = sim_df['COMTRS'].values
    
    a = pm.Normal('a', mu = 0, sigma = 10)
    beta = pm.Normal('beta', mu = 0, sigma = 10)
    year_mu = pm.Normal('year_mu', mu = 0, sigma = 10)
    #year_intercept = pm.Normal('year_intercept', mu = 0, sigma = 10)    
    year_sig = pm.HalfCauchy('year_sig', 3)
    
    year_dum = pm.Normal('year_dum', 
                                mu = 0, sigma = year_sig, shape = 30)
    
    
    
    #plot_mu = pm.Normal('plot_mu', mu =0, sigma =10)
    plot_sig = pm.HalfCauchy('plot_sig', 3)
    
    plot_eff = pm.Normal('plot_eff', mu = 0, sigma = plot_sig, shape = 500)
    
    rho = pm.Normal('rho', mu = 10, sigma =10)
    sigma = pm.HalfCauchy('sigma', 3)
    
    est_LBS_AI = a  + year_dum[year_idx] + rho * treatment*is_certified + beta*treatment
    #est_LBS_AI = a + year_dum[year_idx] + rho * is_certified  + plot_eff[plot]
    #est_LBS_AI = year_idx * year_mu + rho * is_certified + a
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = sigma, 
                           observed =lbs_ai)
    
    #step = pm.Metropolis()
    trace = pm.sample(
                    init = 'advi', 
                      cores = 8,
                      draws = 2000, tune = 500, 
                      #step = step
                      )
#%%
x = trace.get_values('rho')
a_vals = trace.get_values('a')
ymu = trace.get_values('year_mu')
s = trace.get_values('sigma')
b = trace.get_values('beta')
#plot_eff = trace.get_values('plot_eff')

plt.scatter(x, a_vals)
plt.show()
#plt.scatter(x, ymu)
#plt.show()
plt.scatter(x, s)
plt.show()
plt.plot(x)

