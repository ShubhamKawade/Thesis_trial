#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns

bcg = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL_mass.csv')


bcg = general_functions.cleanup(bcg)
bcg = bcg[(bcg['z']>=0.03) & (bcg['z']< 0.15) ]



L_sun = 1
Lbcg = L_sun * 10 ** (0.4*(3.27 - bcg['BCGMag']))
bcg['Lbcg'] = Lbcg

# =============================================================================
# sns.set_context('paper')
# weights = np.ones_like(Lbcg)/len(Lbcg)
# plt.hist(Lbcg/1e11, bins=10, label = 'Normalizations a' )
# #plt.axvline(np.median(T), label='median', color = 'black')
# 
# plt.xlabel(r'$L_{\mathrm{BCG}}$ ($*10^{11}\mathrm{L_{\odot}}$)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,80)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Lbcg_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# 
# =============================================================================
Lx = bcg['Lx']
log_Lx = np.log10(Lx)
sigma_Lx = 0.4343*bcg['eL(%)']/100
err_Lx = bcg['eL(%)']*Lx/100
Lbcg = bcg['Lbcg']

log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6e11
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(sigma_Lx))
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = bcg.sample(n = len(bcg), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = np.empty(len(random_clusters['z']))
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
# 
# 
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_all_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_all_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']
errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)

sns.set_context('paper')
Lbcg_linspace = np.linspace(1,25,100)

z = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept, Slope)
plt.plot(Lbcg_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(Lbcg/1e11,Lx,yerr = err_Lx,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lx)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.ylabel('$L_{\mathrm{X}}$ (*$10^{44}$ erg/s) ')
plt.xlabel('$L_{\mathrm{BCG}}$  (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.title('$L_{\mathrm{X}}-L_{\mathrm{BCG}}$ best fit')
plt.xlim(1,25)
plt.ylim(0.004,60)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-Lbcg_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()
print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')


###################################################

           # Cutting galaxy groups based on mass

################################################
bcg = general_functions.removing_galaxy_groups(bcg)

Lx = bcg['Lx']
log_Lx = np.log10(Lx)
sigma_Lx = 0.4343*bcg['eL(%)']/100
err_Lx = bcg['eL(%)']*Lx/100
Lbcg = bcg['Lbcg']

log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6e11
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(sigma_Lx))
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = bcg.sample(n = len(bcg), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = np.empty(len(random_clusters['z']))
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
# 
# 
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_all(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_all(Mcut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']
errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)

sns.set_context('paper')
Lbcg_linspace = np.linspace(1,25,100)

z = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept_Mcut, Slope_Mcut)
plt.plot(Lbcg_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(Lbcg/1e11,Lx,yerr = err_Lx,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lx)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.ylabel('$L_{\mathrm{X}}$ (*$10^{44}$ erg/s) ')
plt.xlabel('$L_{\mathrm{BCG}}$  (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.title('$L_{\mathrm{X}}-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.xlim(1,25)
plt.ylim(0.004,60)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-Lbcg_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')

#################################################################################
# =============================================================================
# bcgt = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL.txt',sep = '\s+')
# bcgt.rename({'#CLUSTER':'Cluster'},axis=1,inplace=True)
# bcgt = general_functions.cleanup(bcgt)
# 
# bcgt = bcgt[(bcgt['z']>0.03) & (bcgt['z']< 0.15) ]
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# 
# L_sun = 1
# Lbcg = L_sun * 10 ** (0.4*(3.27 - bcgt['BCGMag']))
# bcgt['Lbcg'] = Lbcg
# 
# bcg = pd.merge(bcgt, thesis_table, right_on='Cluster' ,left_on = 'Cluster', how ='inner')
# 
# Lx = bcg['Lx']
# log_Lx = np.log10(Lx)
# sigma_Lx = 0.4343*bcg['eL(%)']/100
# err_Lx = bcg['eL(%)']*Lx/100
# err_Lx.nlargest(3)
# Lbcg = bcg['Lbcg']
# 
# log_Lbcg = np.log10(Lbcg)
# Lbcg_new = Lbcg / 6e11
# log_Lbcg_new = np.log10(Lbcg_new)
# sigma_Lbcg = np.zeros(len(sigma_Lx))
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = bcg['c']/np.median(bcg['c'])
# e_c = bcg['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# cov = np.cov(sigma_Lx,sigma_c)
# 
# ######## Constraing g for concentration #########
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_Lx,sigma_c)
# # for i in g:
# #     yarray = log_Lx - i*log_c
# #     yerr = np.sqrt( (sigma_Lx)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
# #     xarray = log_Lx
# #     xerr = sigma_Lx
# #     
# #     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #     test_scatter.append(Scatter)
# #     test_norm.append(Norm)
# #     test_slope.append(Slope)
# #     gamma.append(i)
# # 
# # test_scatter
# # p = np.where(test_scatter == np.min(test_scatter))
# # P = p[0]
# # test_norm[P[0]],test_slope[P[0]],gamma[P[0]],test_scatter[P[0]]
# # 
# # =============================================================================
# 
# yarray = log_Lx - 0.15*log_c
# xarray = log_Lbcg_new
# yerr = np.sqrt( (sigma_Lx)**2 + (0.15*sigma_c)**2 + 2*0.15*cov[0][1] )
# xerr = sigma_Lbcg 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Lbcg,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($L_{X}/C^{0.15}$-L_{BCG})' )
# plt.errorbar(log_Lbcg,log_Lx,xerr= sigma_Lbcg,yerr = sigma_Lx,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($L_{X}-L_{BCG}$)')
# 
# z = test_Ycept+ test_Slope* log_Lbcg_new
# z1 = ycept + Slope* log_Lbcg_new
# 
# plt.plot(log_Lbcg,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Lbcg,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log($L_{bcg}$)')
# plt.ylabel('$log{Y}$')
# plt.title('$L_{X}/C - L_{BCG}$ scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# =============================================================================


