#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns




r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')
r = general_functions.cleanup(r)

r = r[r['R'] > 2]
R_old = r['R']

bcg = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL.csv')
bcg.rename({'#Cluster':'Cluster'},axis=1,inplace=True)
bcg = general_functions.cleanup(bcg)

bcg = bcg[bcg['z']<0.15]
bcg = bcg[bcg['z']>0.03]
R_new = general_functions.correct_psf(R_old)
Rmin_old = r['R'] - r['Rmin']
Rmin_new = general_functions.correct_psf(Rmin_old)
Rmax_old = r['R'] + r['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

Z = (r['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000
theta = (R_new/60)*np.pi/180
R_kpc = theta * d_A.value
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value
theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
r['R kpc'] = R_kpc
r['Rmin kpc'] = Rmin_kpc
r['Rmax kpc'] = Rmax_kpc
rbcg= pd.merge(bcg, r, right_on='Cluster',left_on = 'Cluster', how ='inner')

# To constrain gamma
# =============================================================================
# re_range = np.arange(-5,5,0.01)
# norm = []
# scatter = []
# slope = []
# re = []
# 
# ## A for loop t run throught each value of re_range and getting Norm, slope and scatter values.
# for i in re_range:
#     
#     Z = (rbcg['z_x'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     R = rbcg['R kpc']
#     R_min = rbcg['Rmin kpc']
#     R_max = rbcg['Rmax kpc']
#     R_new = (R/250)* E**i
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
#     Lbcg = rbcg['L_bcg(1e11solar)']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(Lbcg))
# 
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_Lbcg_new, sigma_Lbcg, log_r, sigma_r)
#     norm.append(Norm)
#     slope.append(Slope)
#     scatter.append(Scatter)
#     re.append(i)
#     
# # To return the index which corresponds to the minimum scatter    
# p = np.where(scatter == np.min(scatter))
# P = p[0]
# re[P[0]]
# =============================================================================
# gamma = -2.04
Z = (rbcg['z_x'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R = rbcg['R kpc']
R_min = rbcg['Rmin kpc']
R_max = rbcg['Rmax kpc']
R_new = (R/250) * E**-2.04
log_r = np.log10(R_new)
err_r = [R-R_min,R_max-R]
sigma_r = 0.4343 * ((R_max-R_min)/(2*R))

Lbcg = rbcg['L_bcg(1e11solar)']
log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(Lbcg))
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_r,sigma_r)



sns.set_context('paper')
Lbcg_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept, Slope)
plt.plot(Lbcg_linspace,z,label='Best fit', color ='green')
plt.errorbar(Lbcg, R, yerr=err_r, ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_Lbcg)})')

plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{BCG}}$ ($10^{11} \,\mathrm{L}_{\odot}$)')
plt.ylabel(' $R*E(z)^{-2.04}$ (kpc)')
plt.title('$R-L_{\mathrm{BCG}}$ best fit')
plt.legend( loc='lower right')
plt.xlim(0.8,35)
plt.ylim(30,2000)
plt.legend(loc='lower right')

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Lbcg_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


# PLotting the residuals
# =============================================================================
# z = ycept + Slope*log_Lbcg_new
# plt.errorbar(R_new,z-log_r,yerr=sigma_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_r)})')
# plt.axhline(0)
# plt.xlabel('R/250 [kpc]')
# plt.ylabel('Residuals')
# #plt.xlim(-1.5,1.2)
# #plt.ylim(1,-1)
# plt.show()
# =============================================================================

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = rbcg.sample(n = len(rbcg), replace = True)
# 
#      omega_m = 0.3
#      omega_lambda = 0.7
#      cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#      Z = (random_clusters['z_x'])
#      E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
# 
#      R = random_clusters['R kpc']
#      R_min = random_clusters['Rmin kpc']
#      R_max = random_clusters['Rmax kpc']
#      R_new = (R/250) * E**-2.04
#      log_r = np.log10(R_new)
#      sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#      Lbcg = random_clusters['L_bcg(1e11solar)']
#      log_Lbcg = np.log10(Lbcg)
#      Lbcg_new = Lbcg / 6
#      log_Lbcg_new = np.log10(Lbcg_new)
#      sigma_Lbcg = np.zeros(len(Lbcg))
# 
#      ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_r,sigma_r)
#      best_A.append(Norm)
#      best_B.append(Slope)
#      best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lbcg_all_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_all_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)
print('The best fit parameters are :')
print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')

###########################################

          # Cutting galaxy groups based on mass

############################################
rbcg = general_functions.removing_galaxy_groups(rbcg)

Z = (rbcg['z_x'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R = rbcg['R kpc']
R_min = rbcg['Rmin kpc']
R_max = rbcg['Rmax kpc']
R_new = (R/250) * E**-2.04
log_r = np.log10(R_new)
err_r = [R-R_min,R_max-R]
sigma_r = 0.4343 * ((R_max-R_min)/(2*R))

Lbcg = rbcg['L_bcg(1e11solar)']
log_Lbcg = np.log10(Lbcg)
Lbcg_new = Lbcg / 6
log_Lbcg_new = np.log10(Lbcg_new)
sigma_Lbcg = np.zeros(len(Lbcg))

ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_r,sigma_r)

sns.set_context('paper')
Lbcg_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept, Slope)
plt.plot(Lbcg_linspace,z,label='Best fit', color ='green')
plt.errorbar(Lbcg, R, yerr=err_r, ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_Lbcg)})')

plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{BCG}}$ ($10^{11} \,\mathrm{L}_{\odot}$)')
plt.ylabel(' $R*E(z)^{-2.04}$ (kpc)')
plt.title('$R-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend( loc='lower right')
plt.xlim(0.8,35)
plt.ylim(30,2000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Lbcg_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = rbcg.sample(n0.0045 = len(rbcg), replace = True)
# 
#      omega_m = 0.3
#      omega_lambda = 0.7
#      cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#      Z = (random_clusters['z_x'])
#      E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
# 
#      R = random_clusters['R kpc']
#      R_min = random_clusters['Rmin kpc']
#      R_max = random_clusters['Rmax kpc']
#      R_new = (R/250) * E**-2.04
#      log_r = np.log10(R_new)
#      sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#      Lbcg = random_clusters['L_bcg(1e11solar)']
#      log_Lbcg = np.log10(Lbcg)
#      Lbcg_new = Lbcg / 6
#      log_Lbcg_new = np.log10(Lbcg_new)
#      sigma_Lbcg = np.zeros(len(Lbcg))
# 
#      ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_r,sigma_r)
#      best_A.append(Norm)
#      best_B.append(Slope)
#      best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lbcg_all(Mcut)_BCES.csv')
# 
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_all(Mcut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)






print(f'Normalization : {np.round(Norm_Mcut ,3)} +/- {np.round(errnorm_Mcut ,3)}')
print(f'Slope : {np.round(Slope_Mcut ,3)} +/- {np.round(errslope_Mcut ,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut ,3)}')












############################################################################################################
# =============================================================================
# r = pd.read_fwf('/home/schubham/Desktop/Half-radii-final.txt',sep ='\\s+')
# r.rename({'# Name':'Cluster'},axis=1,inplace=True)
# r = general_functions.cleanup(r)
# 
# r = r[r['R'] > 2]
# R_old = r['R']
# 
# bcg = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL.txt')
# bcg.rename({'#Cluster':'Cluster'},axis=1,inplace=True)
# bcg = general_functions.cleanup(bcg)
# 
# bcg = bcg[bcg['z']<0.15]
# bcg = bcg[bcg['z']>0.03] 
# 
# R_new = general_functions.correct_psf(R_old)
# Rmin_old = r['R'] - r['Rmin']
# Rmin_new = general_functions.correct_psf(Rmin_old)
# Rmax_old = r['R'] + r['Rmax']
# Rmax_new = general_functions.correct_psf(Rmax_old)
# 
# omega_m = 0.3
# omega_lambda = 0.7
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
# Z = (r['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# d_A = cosmo.angular_diameter_distance(Z)*1000
# theta = (R_new/60)*np.pi/180
# R_kpc = theta * d_A.value
# theta_min = (Rmin_new/60)*np.pi/180
# Rmin_kpc = theta_min * d_A.value
# theta_max = (Rmax_new/60)*np.pi/180
# Rmax_kpc = theta_max * d_A.value
# sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
# r['R kpc'] = R_kpc
# r['Rmin kpc'] = Rmin_kpc
# r['Rmax kpc'] = Rmax_kpc
# rbcg= pd.merge(bcg, r, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# rbcg = pd.merge(rbcg, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# Z = (rbcg['z_x'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# R = rbcg['R kpc']
# R_min = rbcg['Rmin kpc']
# R_max = rbcg['Rmax kpc']
# R_new = (R/250)
# log_r = np.log10(R_new)
# err_r = [(R-R_min)/250,(R_max-R)/250]
# sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# Lbcg = rbcg['L_bcg (1e11 solar)']
# log_Lbcg = np.log10(Lbcg)
# Lbcg_new = Lbcg / 6
# log_Lbcg_new = np.log10(Lbcg_new)
# sigma_Lbcg = np.zeros(len(Lbcg))
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_r,sigma_r)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = rbcg['c']/np.median(rbcg['c'])
# e_c = rbcg['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# cov = np.cov(sigma_r,sigma_c)
# 
# ######## Constraing g for concentration #########
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_r,sigma_c)
# # for i in g:
# #     yarray = log_r - i*log_c
# #     yerr = np.sqrt( (sigma_r)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
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
# yarray = log_r - (-0.420)*log_c
# xarray = log_Lbcg_new
# yerr = np.sqrt( (sigma_r)**2 + (-0.42*sigma_c)**2 + 2*-0.42*cov[0][1] )
# xerr = sigma_Lbcg 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Lbcg,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($R/C^{-0.42}-L_{BCG}$)' )
# plt.errorbar(log_Lbcg,log_r,xerr= sigma_Lbcg,yerr = sigma_r,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($R-L_{BCG}$)')
# 
# z = test_Ycept+ test_Slope* log_Lbcg_new
# z1 = ycept + Slope* log_Lbcg_new
# 
# plt.plot(log_Lbcg,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Lbcg,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log(Lbcg)')
# plt.ylabel('$log{r}$')
# plt.title('$R/c - L_{bcg}$scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# =============================================================================
