#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:03:38 2021

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import time


r = pd.read_fwf('/home/schubham/Desktop/Half-radii-final.txt',sep ='\\s+')
r.rename({'# Name':'Cluster'},axis=1,inplace=True)
r = general_functions.cleanup(r)
rl = r[r['R'] > 2]
R_old = rl['R']
R_new = general_functions.correct_psf(R_old)
Rmin_old = rl['R'] - rl['Rmin']
Rmax_old = rl['R'] + rl['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)
Rmin_new = general_functions.correct_psf(Rmin_old)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (rl['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000

theta = (R_new/60)*np.pi/180
R_kpc = theta * d_A.value

theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
rl['R kpc'] = R_kpc
rl['Rmin kpc'] = Rmin_kpc
rl['Rmax kpc'] = Rmax_kpc


thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rl = pd.merge(rl, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')

gamma_range = np.arange(0,5,0.1)
g_range = np.arange(-0.6,-0.2,0.01)
test_scatter = []
test_norm = []
test_slope = []
gamma = []
gamma_bootstrap = []
g_bootstrap = []
g = []
for k in range(1000):
    random_clusters = rl.sample(n=len(rl), replace = True)
    for i in gamma_range:
        for j in g_range:
            Z = (random_clusters['z'])
            E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
            R = random_clusters['R kpc']
            R_min = random_clusters['Rmin kpc']
            R_max = random_clusters['Rmax kpc']
            R_new = (R/250) * (E**(i))
            log_r = np.log10(R_new)
            sigma_r = 0.4343 * ((R_max-R_min)/(2*R))    
            Lx = random_clusters['Lx']
            log_Lx= np.log10(Lx)
            sigma_Lx = 0.4343*random_clusters['eL']/100
            
            c = random_clusters['c']/np.median(random_clusters['c'])
            e_c = random_clusters['e_c']
            log_c = np.log10(c)
            sigma_c = 0.4343 * e_c/c
            cov = np.cov(sigma_r,sigma_c)
            
            yarray = log_r - j*log_c
            yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
            xarray = log_Lx
            xerr = sigma_Lx
            
            ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
            test_scatter.append(Scatter)
            test_norm.append(Norm)
            test_slope.append(Slope)
            gamma.append(i)
            g.append(j)
    p = np.where(test_scatter == np.min(test_scatter))
    P = p[0]
    gamma_bootstrap.append(gamma[P[0]])
    g_bootstrap.append(g[P[0]])
    print(k, gamma[P[0]],g[P[0]])

bestfit_bootstrap_dict = {'gamma': gamma_bootstrap, 'g': g_bootstrap}              
bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
bestfit_bootstrap.to_csv('R-Lx_gamma&_bootstrap.csv')
np.min(gamma_bootstrap),np.max(gamma_bootstrap)
np.min(g_bootstrap),np.max(g_bootstrap)
plt.hist(g_bootstrap, bins=5)
general_functions.calculate_asymm_err(g_bootstrap)