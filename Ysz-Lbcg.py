#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns



bcgy = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL_mass.csv')
bcgy = general_functions.cleanup(bcgy)
bcgy = bcgy[(bcgy['z']>= 0.03) & (bcgy['z']< 0.15)] 
StN = bcgy['Y']/bcgy['eY']
bcgy = bcgy[ StN > 2]

L_sun = 1
Lbcg = L_sun * 10 ** (0.4*(3.27 - bcgy['BCGMag']))
bcgy['Lbcg'] = Lbcg

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (bcgy['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000
Ysz_arcmin = bcgy['Y']
e_Y_arcmin = bcgy['eY']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz /20
log_Ysz = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
err_Ysz =  ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)

Lbcg = bcgy['Lbcg']
log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6e11
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(Lbcg))
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)

sns.set_context('paper')
Y_linspace = np.linspace(1,25,100)

z = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept, Slope)
plt.plot(Y_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(Lbcg/1e11,Ysz, yerr = err_Ysz,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lbcg)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.xlabel('$L_{\mathrm{BCG}}$ (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.ylabel(' $Y_{\mathrm{SZ}}$  ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{\mathrm{SZ}}-L_{\mathrm{BCG}}$ best fit')
plt.xlim(1.3,25)
plt.ylim(0.5,1000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Ysz-Lbcg_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = bcgy.sample(n = len(bcgy), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
#     Lbcg =random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(Lbcg))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_all_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_all_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']
errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')



#######################################################\
    
    # Cutting galaxy groups based on mass
    
######################################################################
bcgy = general_functions.removing_galaxy_groups(bcgy)
Z = (bcgy['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin = bcgy['Y']
e_Y_arcmin = bcgy['eY']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz /20
log_Ysz = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin

err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)

Lbcg = bcgy['Lbcg']
log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6e11
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(Lbcg))
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)

sns.set_context('paper')
Y_linspace = np.linspace(1,25,100)

z = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept, Slope)
plt.plot(Y_linspace ,z, label = 'Best fit',color = 'green')
plt.errorbar(Lbcg/1e11,Ysz, yerr = err_Ysz,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'Clusters ({len(Lbcg)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.xlabel('$L_{\mathrm{BCG}}$ (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.ylabel(' $Y_{\mathrm{SZ}}$  ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{\mathrm{SZ}}-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')

plt.xlim(1.3,25)
plt.ylim(0.5,1000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Ysz-Lbcg_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = bcgy.sample(n = len(bcgy), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
#     Lbcg =random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(Lbcg))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_all(M_cut)_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_all(M_cut)_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']
errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)


print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')



################################################################################
# =============================================================================
# bcgy = pd.read_fwf('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL.txt')
# bcgy.rename({'#CLUSTER':'Cluster'},axis=1,inplace=True)
# bcgy = general_functions.cleanup(bcgy)
# bcgy = bcgy[(bcgy['z']>0.03) & (bcgy['z']< 0.15)] 
# StN = bcgy['Y']/bcgy['eY']
# bcgy = bcgy[ StN > 2]
# 
# L_sun = 1
# Lbcg = L_sun * 10 ** (0.4*(3.27 - bcgy['BCGMag']))
# bcgy['Lbcg'] = Lbcg
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# 
# bcgy = pd.merge(bcgy, thesis_table, right_on='Cluster' ,left_on = 'Cluster', how ='inner')
# 
# Z = (bcgy['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
# Ysz_arcmin = bcgy['Y']
# e_Y_arcmin = bcgy['eY']
# Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
# Ysz_new = Ysz /35
# log_Ysz = np.log10(Ysz_new)
# sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)/35
# 
# Lbcg = bcgy['Lbcg']
# log_Lbcg = np.log10(Lbcg)
# Lbcg_new = Lbcg / 6e11
# log_Lbcg_new = np.log10(Lbcg_new)
# sigma_Lbcg = np.zeros(len(Lbcg))
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = bcgy['c']/np.median(bcgy['c'])
# e_c = bcgy['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# cov = np.cov(sigma_Ysz,sigma_c)
# 
# ######## Constraing g for concentration #########
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_Ysz,sigma_c)
# # for i in g:
# #     yarray = log_Ysz - i*log_c
# #     yerr = np.sqrt( (sigma_Ysz)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
# #     xarray = log_Lbcg_new
# #     xerr = sigma_Lbcg
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
# yarray = log_Ysz - (-0.01)*log_c
# xarray = log_Lbcg_new
# yerr = np.sqrt( (sigma_Ysz)**2 + (-0.01*sigma_c)**2 + 2*(-0.01)*cov[0][1] )
# xerr = sigma_Lbcg 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Lbcg,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($Y_{SZ}/C^{-0.01}$-L_{BCG})' )
# plt.errorbar(log_Lbcg,log_Ysz,xerr= sigma_Lbcg,yerr = sigma_Ysz,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($Y_{SZ}-L_{BCG}$)')
# 
# z = test_Ycept+ test_Slope* log_Lbcg_new
# z1 = ycept + Slope* log_Lbcg_new
# 
# plt.plot(log_Lbcg,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Lbcg,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log($L_{BCG}$)')
# plt.ylabel('$log{Y}$')
# plt.title('$Y_{SZ}/C - L_{BCG}$ scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# =============================================================================


