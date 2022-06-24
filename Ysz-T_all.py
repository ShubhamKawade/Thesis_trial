#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
import seaborn as sns
from astropy.cosmology import LambdaCDM 

sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')
sz = general_functions.cleanup(sz)

# Filter using S/N ratio
StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
sz = sz[StN >= 2]
omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (sz['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = sz['e_Y']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)


# =============================================================================
# sns.set_context('paper')
# weights = np.ones_like(Ysz)/len(Ysz)
# plt.hist(Ysz, bins=8, label = 'Normalizations a' )
# #plt.axvline(np.median(T), label='median', color = 'black')
# 
# plt.xlabel(r'$Y_{\mathrm{SZ}}$ ($\mathrm{kpc^{2}}$)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,260)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Ysz_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# =============================================================================



Ysz_new = Ysz * E/35
log_Ysz = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin

T = sz['T(keV)']
T_new = T/4.5
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_T = 0.4343*((sz['Tmax']-sz['Tmin'])/(2*T))
err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E
err_T = [T - sz['Tmin'], sz['Tmax']-T]
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)

# Plotting the best fit 
# =============================================================================
# sns.set_context('paper')
# plt.errorbar(T,Ysz_new,xerr= err_T,yerr = err_Ysz,color = 'red',ls='', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'Clusters ({len(T_new)})' )
# z = Norm * T_new ** Slope
# plt.plot(T,z, color = 'green',label = 'bestfit')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$T$ (keV)')
# plt.ylabel('$Y_{\mathrm{sz}}*E(z)$ ($35^{-1} \,\mathrm{kpc}^{2}$)')
# plt.title('$Y_{SZ}-T$ best fit')
# plt.legend(loc='lower right')
# plt.xlim(0.6,25)
# plt.ylim(0.003,90)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Ysz-T_bestfit.png',dpi=300,bbox_inches="tight")
# plt.show()
# =============================================================================

T_linspace = np.linspace(0.5,100,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept, Slope)
#plt.plot(T_linspace, z)

plt.errorbar(T,Ysz*E,xerr= err_T,yerr = err_Ysz,color = 'red',ls='', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'clusters ({len(T_new)})' )
plt.plot(T_linspace,z, color = 'green',label = 'bestfit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T$ (keV)')
plt.ylabel('$Y_{\mathrm{sz}}*E(z)$ ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{SZ}-T$ best fit')
plt.legend(loc='lower right')
plt.xlim(0.8,30)
plt.ylim(0.5,1500)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Ysz-T_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap using BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = sz.sample(n = len(sz), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
#     Ysz_arcmin = random_clusters['Y(r/no_ksz,arcmin^2)']
#     e_Y_arcmin = random_clusters['e_Y']
#     Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz * E/35
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     log_T = np.log10(T)
#     log_T_new = np.log10(T_new)
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
#     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_all_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_all_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']


errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')





#####################################

    # Cutting galaxy groups based on mass

##################################


sz = general_functions.removing_galaxy_groups(sz)
# Filter using S/N ratio
StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
sz = sz[StN >= 2]
omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (sz['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = sz['e_Y']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz * E/35
log_Ysz = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin

T = sz['T(keV)']
T_new = T/4.5
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_T = 0.4343*((sz['Tmax']-sz['Tmin'])/(2*T))
err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E
err_T = [T - sz['Tmin'], sz['Tmax']-T]
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)

# Plotting the best fit 
sns.set_context('paper')
T_linspace = np.linspace(0.5,100,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept_Mcut, Slope_Mcut)

plt.errorbar(T,Ysz*E,xerr= err_T,yerr = err_Ysz,color = 'red',ls='', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'clusters ({len(T_new)})' )
#z = Norm_Mcut * T_new ** Slope_Mcut
plt.plot(T_linspace,z, color = 'green',label = 'bestfit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('T (keV)')
plt.ylabel(r'$Y_{sz}*E(z)$ ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{SZ}-T$ best fit ($M_{cluster} > 10^{14}M_{\odot}$)')
plt.legend(loc='lower right')
plt.xlim(0.8,30)
plt.ylim(0.5,1500)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Ysz-T_Mcut_bestfit.png',bbox_inches="tight")
plt.show()






# Bootstrap using BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = sz.sample(n = len(sz), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
#     Ysz_arcmin = random_clusters['Y(r/no_ksz,arcmin^2)']
#     e_Y_arcmin = random_clusters['e_Y']
#     Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz * E/35
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     log_T = np.log10(T)
#     log_T_new = np.log10(T_new)
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
#     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_all(Mcut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_all(Mcut)_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']


errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)




print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')











# =============================================================================
# ######### To include the concentration parameter in the saling relation #####
# sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_new.txt', sep ='\s+')
# sz = sz.rename({'#Cluster':'Cluster'},axis=1)
# sz = general_functions.cleanup(sz)
# StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
# sz = sz[StN >2]
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# sz = pd.merge(sz, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# ######## OLD SCALING RELATION VARIABLES #################
# omega_m = 0.3
# omega_lambda = 0.7
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# Z = (sz['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# D_a = cosmo.angular_diameter_distance(Z) * 1000
# Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
# e_Y_arcmin = sz['e_Y']
# Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
# Ysz_new = Ysz * E/35
# log_Ysz = np.log10(Ysz_new)
# sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# Ysz.median()
# T = sz['T(keV)']
# T_new = T/5
# log_T = np.log10(T)
# log_T_new = np.log10(T_new)
# sigma_T = 0.4343*((sz['Tmax']-sz['Tmin'])/(2*T))
# err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)/35
# err_T = [T - sz['Tmin'], sz['Tmax']-T]
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
# ############## Adding new variables ##################3
# c = sz['c']/np.median(sz['c'])
# e_c = sz['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# ### To constrain the g parameter for concentration ####################
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_ycept = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_Ysz,sigma_c)
# # for i in g:
# #     yarray = log_Ysz - i*log_c
# #     yerr = np.sqrt( (sigma_Ysz)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
# #     xarray = log_T_new
# #     xerr = sigma_T
# #     
# #     test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #     test_ycept.append(test_Ycept)
# #     test_scatter.append(test_Scatter)
# #     test_norm.append(test_Norm)
# #     test_slope.append(test_Slope)
# #     gamma.append(i)
# # 
# # test_scatter
# # p = np.where(test_scatter == np.min(test_scatter))
# # P = p[0]
# # test_norm[P[0]],test_slope[P[0]],gamma[P[0]],test_scatter[P[0]]
# # =============================================================================
# 
# ########## Performing best fit for the new relation #################
# cov = np.cov(sigma_Ysz,sigma_c)
# yarray = log_Ysz - 0.099*log_c
# xarray = log_T_new
# yerr = np.sqrt( (sigma_Ysz)**2 + (0.099*sigma_c)**2 - 2*0.099*cov[0][1])
# xarray = log_T_new
# xerr = sigma_T 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_T,yarray,xerr= sigma_T,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($Y_{sz}/C^{0.099}$-T)' )
# plt.errorbar(log_T,log_Ysz,xerr= sigma_T,yerr = sigma_Ysz,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($Y_{SZ}-T$)')
# 
# z = test_Ycept+ test_Slope* log_T_new
# z1 = ycept + Slope* log_T_new
# 
# plt.plot(log_T,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_T,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log(T/keV)')
# plt.ylabel('$log{Y}$')
# plt.title('$Y_{SZ}$/C - T scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# ################################################################################
#                           # To constain gamma an g simultaneously
# ##############################################################################3##
# # =============================================================================
# # gamma_range = np.arange(-7,-5,0.01)
# # g_range = np.arange(-2,2,0.01)
# # 
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # g = []
# # cov = np.cov(sigma_Ysz,sigma_c)
# # for i in gamma_range:
# #     for j in g_range:
# #         Z = (sz['z'])
# #         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #         D_a = cosmo.angular_diameter_distance(Z) * 1000
# #         Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
# #         e_Y_arcmin = sz['e_Y']
# #         Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
# #         Ysz_new = Ysz * E**i/35
# #         log_Ysz = np.log10(Ysz_new)
# #         sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# #         Ysz.median()
# #         T = sz['T(keV)']
# #         T_new = T/5
# #         log_T = np.log10(T)
# #         log_T_new = np.log10(T_new)
# #         sigma_T = 0.4343*((sz['Tmax']-sz['Tmin'])/(2*T))
# #                 
# #         c = sz['c']/np.median(sz['c'])
# #         e_c = sz['e_c']
# #         log_c = np.log10(c)
# #         sigma_c = 0.4343 * e_c/c
# #         
# #         
# #         cov = np.cov(sigma_Ysz,sigma_c)
# #         yarray = log_Ysz - j*log_c
# #         yerr = np.sqrt( (sigma_Ysz)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
# #         xarray = log_T_new
# #         xerr = sigma_T
# #         ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #         test_scatter.append(Scatter)
# #         test_norm.append(Norm)
# #         test_slope.append(Slope)
# #         gamma.append(i)
# #         g.append(j)
# # 
# # p = np.where(test_scatter == np.min(test_scatter))
# # P = p[0]
# # test_scatter[P[0]],gamma[P[0]],g[P[0]]
# # 
# # =============================================================================
# 
# =============================================================================
