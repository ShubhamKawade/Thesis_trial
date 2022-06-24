#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:40:50 2021

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 





r = pd.read_fwf('/home/schubham/Thesis/Thesis/Data/Half-radii-T-NEW-2.txt',sep ='\\s+')
r.rename({'#Name':'Cluster'},axis=1,inplace=True)
r = general_functions.cleanup(r)
rt = r[r['R'] > 2]

R_old = rt['R']
R_new = general_functions.correct_psf(R_old)
Rmin_old = rt['R'] - rt['Rmin']
Rmax_old = rt['R'] + rt['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)
Rmin_new = general_functions.correct_psf(Rmin_old)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (rt['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000

theta = (R_new/60)*np.pi/180
R_kpc = theta * d_A.value

theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
rt['R kpc'] = R_kpc
rt['Rmin kpc'] = Rmin_kpc
rt['Rmax kpc'] = Rmax_kpc

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rt = pd.merge(rt, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')

# Bootstrap for gamma and g
# =============================================================================
# re_range = np.arange(-2,3,0.1)
# g_range = np.arange(-0.6,-0.2,0.01)
# 
# test_scatter = []
# gamma_bootstrap = []
# g_bootstrap = []
# gamma = []
# g = []
# for k in range(1000):
#     random_clusters = rt.sample(n = len(rt), replace = True)
#     
#     for i in re_range:
#         for j in g_range:
#             Z = (random_clusters['z'])
#             E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#             
#             R = random_clusters['R kpc']
#             R_min = random_clusters['Rmin kpc']
#             R_max = random_clusters['Rmax kpc']
#             R_new = (R/250) * (E**(i))
#             log_r = np.log10(R_new)
#             sigma_r = 0.4343 * (R_max-R_min)/(2*R)
#             
#             T = random_clusters['T']
#             T_new = T/4.5
#             log_T = np.log(T)
#             log_T_new = np.log10(T/4.5)
#             sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#             
#             c = random_clusters['c']/np.median(random_clusters['c'])
#             e_c = random_clusters['e_c']
#             log_c = np.log10(c)
#             sigma_c = 0.4343 * e_c/c
#             cov = np.cov(sigma_r,sigma_c)
#             
#             yarray = log_r - j*log_c
#             yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#             xarray = log_T_new
#             xerr = sigma_T
#             
#             ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#             test_scatter.append(Scatter)
#             gamma.append(i)
#             g.append(j)
#     p = np.where(test_scatter == np.min(test_scatter))
#     P = p[0]
#     gamma_bootstrap.append(gamma[P[0]])
#     g_bootstrap.append(g[P[0]])
#     print(k,gamma[P[0]],g[P[0]])
# 
# bestfit_bootstrap_dict = {'gamma': gamma_bootstrap, 'g': g_bootstrap}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_gamma_g_bootstrap.csv')
# plt.hist(gamma_bootstrap, bins =8 )
# plt.show()
# plt.hist(g_bootstrap, bins=range(min(g_bootstrap), max(g_bootstrap) + binwidth, binwidth))
# plt.xlim(-0.40, -0.37)
# plt.show()
# =============================================================================
