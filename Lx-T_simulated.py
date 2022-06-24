#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:59:59 2022

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
import seaborn as sns
from astropy.cosmology import LambdaCDM 



# Best fit values for the real Lx-T fit :
Norm_real = 1.421
err_Norm_real = 0.001
Slope_real = 2.10
err_Slope_real = 0.003
Scatter_real = 0.239
err_scatter_real = 0.001

Norm_Mcut_real = 1.575
err_Norm_Mcut_real = 0.003
Slope_Mcut_real = 1.882
err_Slope_Mcut_real = 0.007
Scatter_Mcut_real = 0.207
err_scatter_Mcut_real = 0.001


Cluster = []
Norm = []
Slope = []
Scatter = []
Norm_Mcut = []
Slope_Mcut = []
Scatter_Mcut = []

for j in range(1,1001,1):
    print(j)
    
    sample = pd.read_csv(f'/home/schubham/Thesis/Thesis/Simulated_clusters/simulated_clusters_with_mass/sample_{j}.csv')
    
        
    omega_m = 0.3
    omega_lambda = 0.7
    Z = (sample['z'])
    np.median(Z)
    Lx = sample['Lx']
    np.median(Lx)
    E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
    Lx_new = sample['Lx']
    T = sample['T']
    T_new = T/4.5
    
    log_Lx = np.log10(Lx_new)
    log_T = np.log10(T)
    log_T_new = np.log10(T_new)
    sigma_Lx = 0.4343*sample['eL']/100
    sigma_T = 0.4343*((sample['Tmax']-sample['Tmin'])/(2*T))
    
    err_Lx = sample['eL']*Lx_new/100
    err_T = [(T-sample['Tmin']), (sample['Tmax']-T)]
    
    ycept, norm, slope, scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
    
    
    ############## CUTTING THE GALAXY GROUPS #######################
    
    
    sample = general_functions.removing_galaxy_groups(sample)
    
    omega_m = 0.3
    omega_lambda = 0.7
    Z = (sample['z'])
    Lx = sample['Lx']
    E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
    Lx_new = sample['Lx']
    T = sample['T']
    T_new = T/4.5
    
    log_Lx = np.log10(Lx_new)
    log_T = np.log10(T)
    log_T_new = np.log10(T_new)
    sigma_Lx = 0.4343*sample['eL']/100
    sigma_T = 0.4343*((sample['Tmax']-sample['Tmin'])/(2*T))
    
    err_Lx = sample['eL']*Lx_new/100
    err_T = [(T-sample['Tmin']), (sample['Tmax']-T)]
    
    ycept_Mcut, norm_Mcut, slope_Mcut, scatter_Mcut = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
    
    Cluster.append(sample['Cluster'])
    Norm.append(norm)
    Slope.append(slope)
    Scatter.append(scatter)
    Norm_Mcut.append(norm_Mcut)
    Slope_Mcut.append(slope_Mcut)
    Scatter_Mcut.append(scatter_Mcut)

data = {'Sample' : [Cluster], 'Norm_sim' : [Norm]}
plt.hist(Norm, bins=10 )
bias_Norm = (np.median(Norm) - np.median(Norm_Mcut))/np.median(Norm) * 100
