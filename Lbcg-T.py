#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns

# =============================================================================
# bcgt = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL.csv')
# mass = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_masses.csv')
# 
# bcgt = general_functions.cleanup(bcgt)
# bcgt_mass = pd.merge(bcgt, mass, left_on = bcgt['Cluster'].str.casefold(), right_on = mass['Cluster'].str.casefold(), how = 'inner')
# bcgt_mass.to_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL_mass.csv') 
# 
# =============================================================================
bcgt = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL_mass.csv')
bcgt = general_functions.cleanup(bcgt)

bcgt = bcgt[(bcgt['z']>=0.03) & (bcgt['z']< 0.15) ]

T = bcgt['T']
T_new = T/4.5
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_T = 0.4343*((bcgt['T+']-bcgt['T-'])/(2*T))
#sigma_T=np.zeros(len(sigma_T))

Lbcg = bcgt['L_bcg(1e11solar)']
Lbcg_new = Lbcg / 6
log_Lbcg = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(sigma_T))
err_T = [T-bcgt['T-'], bcgt['T+']-T]
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Lbcg,sigma_Lbcg)

sns.set_context('paper')
T_linspace = np.linspace(1,25,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 6, ycept, Slope)
plt.plot(T_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(T,Lbcg, xerr = err_T,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lbcg)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.ylabel('$L_{\mathrm{BCG}}$ ($\mathrm{L}_{\odot}$)')
plt.xlabel(' $T$ (keV)')
plt.title('$L_{\mathrm{BCG}}-T$ best fit')
plt.xlim(1.,20)
plt.ylim(0.5,50)

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lbcg-T_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


# Bootstrap  : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = bcgt.sample(n = len(bcgt), replace = True)
#     
#     T =random_clusters ['T']
#     T_new = T/4.5
#     log_T = np.log10(T)
#     log_T_new = np.log10(T_new)
#     sigma_T = 0.4343*((random_clusters ['T+']-random_clusters ['T-'])/(2*T))
#     Lbcg = random_clusters['L_bcg(1e11solar)']
#     Lbcg_new = Lbcg / 6
#     log_Lbcg = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_T))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Lbcg,sigma_Lbcg)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lbcg-T_all_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lbcg-T_all_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')


####################################################

      # Cutting galaxy groups based on mass

####################################################

bcgt = general_functions.removing_galaxy_groups(bcgt)
T = bcgt['T']
T_new = T/4.5
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_T = 0.4343*((bcgt['T+']-bcgt['T-'])/(2*T))
#sigma_T=np.zeros(len(sigma_T))

Lbcg = bcgt['L_bcg(1e11solar)']
Lbcg_new = Lbcg / 6
log_Lbcg = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(sigma_T))
err_T = [T-bcgt['T-'], bcgt['T+']-T]
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Lbcg,sigma_Lbcg)

sns.set_context('paper')
T_linspace = np.linspace(1,25,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 6, ycept, Slope)
plt.plot(T_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(T,Lbcg, xerr = err_T,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lbcg)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.ylabel('$L_{\mathrm{BCG}}$ ($\mathrm{L}_{\odot}$)')
plt.xlabel(' $T$ (keV)')
plt.title('$L_{\mathrm{BCG}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.xlim(1.,20)
plt.ylim(0.5,50)

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lbcg-T_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap  : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = bcgt.sample(n = len(bcgt), replace = True)
#     
#     T =random_clusters ['T']
#     T_new = T/4.5
#     log_T = np.log10(T)
#     log_T_new = np.log10(T_new)
#     sigma_T = 0.4343*((random_clusters ['T+']-random_clusters ['T-'])/(2*T))
#     Lbcg = random_clusters['L_bcg(1e11solar)']
#     Lbcg_new = Lbcg / 6
#     log_Lbcg = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_T))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Lbcg,sigma_Lbcg)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lbcg-T_all(Mcut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lbcg-T_all(Mcut)_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')



###############################################################################
# =============================================================================
# bcgt = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL.txt')
# bcgt.rename({'#Cluster':'Cluster'},axis=1,inplace=True)
# bcgt = general_functions.cleanup(bcgt)
# bcgt = bcgt[(bcgt['z']>0.03) & (bcgt['z']< 0.15) ]
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# bcgt = pd.merge(bcgt, thesis_table, right_on='Cluster' ,left_on = 'Cluster', how ='inner')
# 
# T = bcgt['T']
# T_new = T/4.3
# log_T = np.log10(T)
# log_T_new = np.log10(T_new)
# sigma_T = 0.4343*((bcgt['T+']-bcgt['T-'])/(2*T))
# #sigma_T=np.zeros(len(sigma_T))
# 
# Lbcg = bcgt['L_bcg (1e11 solar)']
# Lbcg_new = Lbcg / 6
# log_Lbcg = np.log10(Lbcg_new)
# sigma_Lbcg = np.zeros(len(sigma_T))
# err_T = [T-bcgt['T-'], bcgt['T+']-T]
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Lbcg,sigma_Lbcg)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = bcgt['c']/np.median(bcgt['c'])
# e_c = bcgt['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# cov = np.cov(sigma_Lbcg,sigma_c)
# 
# ######## Constraing g for concentration #########
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_Lbcg,sigma_c)
# # for i in g:
# #     yarray = log_Lbcg - i*log_c
# #     yerr = np.sqrt( (sigma_Lbcg)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
# #     xarray = log_T_new
# #     xerr = sigma_T
# #     
# #     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #     test_scatter.append(Scatter)
# #     test_norm.append(Norm)
# #     test_slope.append(Slope)
# #     gamma.append(i)
# # 
# # p = np.where(test_scatter == np.min(test_scatter))
# # P = p[0]
# # test_norm[P[0]],test_slope[P[0]],gamma[P[0]],test_scatter[P[0]]
# # 
# # =============================================================================
# cov = np.cov(sigma_Lbcg,sigma_c)
# yarray = log_Lbcg - (0.039)*log_c 
# xarray = log_T_new
# yerr = np.sqrt( (sigma_Lbcg)**2 + (0.039*sigma_c)**2 - 2*(0.039)*cov[0][1] )
# xerr = sigma_T 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_T,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($L_{BCG}/C^{0.039}$-T)' )
# plt.errorbar(log_T,log_Lbcg,xerr= sigma_T,yerr = sigma_Lbcg,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($L_{BCG}-T$)')
# 
# z = test_Ycept+ test_Slope* log_T_new
# z1 = ycept + Slope* log_T_new
# 
# plt.plot(log_T,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_T,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log($T$)')
# plt.ylabel('$log{Y}$')
# plt.title('$L_{BCG}/C - T$ scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# =============================================================================


