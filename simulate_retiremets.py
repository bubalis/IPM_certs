#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:41:30 2020

@author: bdube
"""
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from crop_field_setup import save_shape_w_cols, load_shape_w_cols, crop_shp_namer