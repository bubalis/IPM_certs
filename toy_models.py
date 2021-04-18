#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:31:28 2021

@author: bdube
"""

#%%
year_sum = subset.groupby('YEAR').sum()[['ACRES_PLANTED', "LBS_AI"]]
year_sum['LBS_per_ac'] = year_sum['LBS_AI']/year_sum['ACRES_PLANTED']

year_sum.reset_index(inplace = True)
years = year_sum.index


N_KNOT = 10
knots = np.linspace(years.min(), years.max(), N_KNOT+1)[1:]
basis_funcs = sp.interpolate.BSpline(knots, np.eye(N_KNOT), k=1)
trend_x = basis_funcs(year_sum.index)
trend_x_ = shared(trend_x)

with pm.Model() as model:
    lbs_ai = year_sum['LBS_per_ac']
    
    σ_y = pm.HalfCauchy('σ_y', 5)
    y0 = pm.Normal('y0', 0, 10)
    Δ_y = pm.Normal('Δ_y', 0, 1, shape=N_KNOT)
    y = pm.Deterministic('y', y0 + (σ_y * Δ_y).cumsum())
    
    sigma = pm.HalfCauchy('sig', 3)
    
    est_LBS_AI = trend_x_.dot(y)
    
    likelihood = pm.Normal('likelihood',  
                           mu = est_LBS_AI, 
                           sigma = sigma, 
                           observed = lbs_ai)
        
    trace = pm.sample(
                init = 'advi', 
                  cores = 8,
                  draws = 1500, tune = 1500, 
                  )
    
