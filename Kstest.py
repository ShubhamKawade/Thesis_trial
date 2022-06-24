#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:56:39 2021

@author: schubham
"""
import pandas as pd
import general_functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

master_file = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_new.txt', sep = '\\s+')
LBY_full_ee = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL.txt',sep = '\\s+')
half_radii_final = pd.read_fwf('/home/schubham/Thesis/Thesis/Data/Half-radii-final.txt',sep = '\\s+')
thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
offset = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_FINAL_ANISOTROPY_BCG_OFFSET.csv',sep = '\t')
temp_data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/temperatures_inner_outer.csv')

temp_data = temp_data.rename({'#Name':'Cluster'},axis=1)
temp_data = general_functions.cleanup(temp_data)
temp_data.iloc[0]
offset = offset.rename({'#Cluster':'Cluster'},axis=1)
offset = general_functions.cleanup(offset)

master_file = master_file.rename({'#Cluster':'Cluster'},axis=1)
master_file = general_functions.cleanup(master_file)

LBY_full_ee = LBY_full_ee.rename({'#CLUSTER':'Cluster'},axis=1)
LBY_full_ee = general_functions.cleanup(LBY_full_ee)

half_radii_final = half_radii_final.rename({'#Name':'Cluster'},axis=1)
half_radii_final = general_functions.cleanup(half_radii_final)

thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)

temp_data.iloc[0]
offset.iloc[0]

# Using the half-light-radii-file for nHtot
data_all = pd.merge(temp_data, thesis_table, left_on='Cluster', right_on='Cluster', how='inner')
data_all_new = pd.merge(temp_data, offset, left_on='Cluster', right_on='Cluster', how='inner')
#data_all = pd.merge(offset, thesis_table, left_on='Cluster', right_on='Cluster', how='inner')
g = data_all.groupby('label')
CC = g.get_group('CC')
NCC = g.get_group('NCC')
re = data_all_new[data_all_new['BCG_offset_R500'] < 0.01]
de = data_all_new[data_all_new['BCG_offset_R500'] > 0.08]

x = re['T']
y = de['T']

a,b,c   = plt.hist(x, bins=10, cumulative=True, histtype='step', density = True, label = f're ({len(x)})', color = 'green', alpha = 0.9)
plt.hist(y, bins=10, cumulative=True, histtype='step', density = True, label = f'de ({len(y)})', color = 'red', alpha = 0.9)
plt.axvline(np.median(x), color = 'blue', label = 're med')
plt.axvline(np.median(y), color = 'orange', label = 'de med')

print(a)
# =============================================================================
# bins_c, cdf_c = general_functions.calculate_cdf(Nh_c, 20)
# bins_n, cdf_n = general_functions.calculate_cdf(Nh_n, 20)
# plt.plot(bins_c[1:], cdf_c, label = f'CC ({len(CC)})')
# plt.plot(bins_n[1:], cdf_n, label = f'NCC ({len(NCC)})')
# =============================================================================
#plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('T [keV]' )
#plt.xlim(1,13)
plt.ylabel('CDF')
plt.title('CDF for T')
#plt.savefig('/home/schubham/Thesis/Thesis/Scaling_relations/py_scripts/CDF_Nhtot_ccVncc.png',dpi = 300)
general_functions.calculate_ks_stat(x,y)

data = pd.merge(offset, thesis_table, left_on='Cluster', right_on='Cluster', how='inner')
concentration = data['c']
err_concentration = data['e_c']
xbcg_offset = data['BCG_offset_R500']
plt.errorbar(xbcg_offset, concentration, yerr=err_concentration, ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(offset)})')
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.006,2)
plt.xlim(0.0005,4)
plt.ylabel('Concentration')
plt.xlabel('Xray-BCG-offset')
plt.savefig('/home/schubham/Desktop/concVoffset.png',dpi = 300)
