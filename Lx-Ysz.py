#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns

sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')
sz = general_functions.cleanup(sz)
StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
sz = sz[StN > 2]

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (sz['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = sz['e_Y']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz/35
log_Ysz = np.log10(Ysz)
log_Ysz_new = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
Lx = sz['Lx(1e44)']
Lx_new = Lx * (E)**(-5/3) #### The power of redshift evolution for this relations is -5/3
log_Lx = np.log10(Lx_new)
sigma_Lx = 0.4343*sz['eL(%)']/100
err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)
err_Lx = sz['eL(%)']*Lx_new/100
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_Lx,sigma_Lx)




# =============================================================================
# 
# z = ycept + Slope * log_Ysz_new
# 
# ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,z-log_Lx,sigma_Lx)
# z_n = ycept_n + Slope_n * log_Ysz_new
# 
# plt.plot(log_Ysz,z_n,ls ='solid',color = 'black')
# plt.errorbar(log_Ysz,z-log_Lx,yerr=sigma_Lx,color = 'red',ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'clusters ({len(Lx)})' )
# plt.axhline(0)
# plt.ylim(-1,1)
# plt.ylabel('$L_{X}*E(z)^{-5/3}/10^{44}$ erg/s ')
# plt.xlabel(' $Y_{SZ}/60$ $kpc^{2}$')
# plt.legend()
# plt.show()
# =============================================================================

# Bootstrap using BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = sz.sample(n = len(sz), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin = random_clusters['Y(r/no_ksz,arcmin^2)']
#     e_Y_arcmin = random_clusters['e_Y']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /35
#     log_Ysz = np.log10(Ysz)
#     log_Ysz_new = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
#     Lx = random_clusters['Lx(1e44)']
#     Lx_new = Lx * (E)**(-5/3) #### The power of redshift evolution for this relations is -5/3
#     log_Lx = np.log10(Lx_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_Lx,sigma_Lx)
# 
#   
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Ysz_all_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Ysz_all_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm =  general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter =  general_functions.calculate_asymm_err(scatter)
#z = Norm * Ysz_new ** Slope
sns.set_context('paper')
Y_linspace = np.linspace(0.1,2000,100)

z = general_functions.plot_bestfit(Y_linspace, 35, 1, ycept, Slope)
plt.plot(Y_linspace,z, label = 'Best fit',color = 'green')
plt.errorbar(Ysz,Lx_new,yerr=err_Lx,xerr= err_Ysz,color = 'red',ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'Clusters ({len(Lx)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.ylabel('$L_{\mathrm{X}}*E(z)^{-5/3} \,(*10^{44}$ erg/s) ')
plt.xlabel(' $Y_{\mathrm{SZ}}$ ($\mathrm{kpc}^{2})$')
plt.title('$L_{\mathrm{X}}-Y_{\mathrm{SZ}}$ best fit')
plt.xlim(0.3,1500)
plt.ylim(0.01,50)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-Ysz_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')



################# CUTTING GALAXY GROUPS ############################
sz = general_functions.removing_galaxy_groups(sz)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (sz['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = sz['e_Y']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)

Ysz_new = Ysz/35
log_Ysz = np.log10(Ysz)
log_Ysz_new = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin

Lx = sz['Lx(1e44)']
Lx_new = Lx * (E)**(-5/3) #### The power of redshift evolution for this relations is -5/3
log_Lx = np.log10(Lx_new)
sigma_Lx = 0.4343*sz['eL(%)']/100
err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)
err_Lx = sz['eL(%)']*Lx_new/100
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_Lx,sigma_Lx)





# =============================================================================
# z = ycept + Slope * log_Ysz_new
# 
# ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,z-log_Lx,sigma_Lx)
# z_n = ycept_n + Slope_n * log_Ysz_new
# 
# plt.plot(log_Ysz,z_n,ls ='solid',color = 'black')
# plt.errorbar(log_Ysz,z-log_Lx,yerr=sigma_Lx,color = 'red',ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'clusters ({len(Lx)})' )
# plt.axhline(0)
# plt.ylim(-1,1)
# plt.ylabel('$L_{X}*E(z)^{-5/3}/10^{44}$ erg/s ')
# plt.xlabel(' $Y_{SZ}/60$ $kpc^{2}$')
# plt.legend()
# plt.show()
# =============================================================================
# Bootstrap using BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = sz.sample(n = len(sz), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin = random_clusters['Y(r/no_ksz,arcmin^2)']
#     e_Y_arcmin = random_clusters['e_Y']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /35
#     log_Ysz = np.log10(Ysz)
#     log_Ysz_new = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
#     Lx = random_clusters['Lx(1e44)']
#     Lx_new = Lx * (E)**(-5/3) #### The power of redshift evolution for this relations is -5/3
#     log_Lx = np.log10(Lx_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_Lx,sigma_Lx)
# 
#   
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Ysz_all(M_cut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Ysz_all(M_cut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)

sns.set_context('paper')
Y_linspace = np.linspace(0.1,2000,100)

z = general_functions.plot_bestfit(Y_linspace, 35, 1, ycept_Mcut, Slope_Mcut)
plt.plot(Y_linspace,z, label = 'Best fit',color = 'green')
plt.errorbar(Ysz,Lx_new,yerr=err_Lx,xerr= err_Ysz,color = 'red',ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'Clusters ({len(Lx)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.ylabel('$L_{\mathrm{X}}*E(z)^{-5/3} \,(*10^{44}$ erg/s) ')
plt.xlabel(' $Y_{\mathrm{SZ}}$ ($\mathrm{kpc}^{2})$')
plt.title('$L_{X}-Y_{SZ}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.xlim(0.3,1500)
plt.ylim(0.01,50)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-Ysz_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()



print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')






############# Scaling relation including concentration parameter ##################
# =============================================================================
# sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_new.txt', sep = '\\s+')
# sz = sz.rename({'#Cluster':'Cluster'},axis=1)
# sz = general_functions.cleanup(sz)
# StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
# sz = sz[StN > 2]
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# sz = pd.merge(sz, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# 
# Z = (sz['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# D_a = cosmo.angular_diameter_distance(Z) * 1000
# Ysz_arcmin = sz['Y(r/no_ksz,arcmin^2)']
# e_Y_arcmin = sz['e_Y']
# Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
# Ysz_new = Ysz  /60
# log_Ysz = np.log10(Ysz)
# log_Ysz_new = np.log10(Ysz_new)
# sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# Lx = sz['Lx(1e44)']
# Lx_new = Lx * (E)**(-5/3) #### The power of redshift evolution for this relations is -5/3
# log_Lx = np.log10(Lx_new)
# sigma_Lx = 0.4343*sz['eL(%)']/100
# err_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)
# err_Lx = sz['eL(%)']*Lx_new/100
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_Lx,sigma_Lx)
# 
# ########### Adding a new variable ##################
# c = sz['c']/np.median(sz['c'])
# e_c = sz['e_c']
# log_c = np.log10(c)
# sigma_c = 0.4343 * e_c/c
# 
# ### To constrain the g parameter for concentration ####################
# 
# # =============================================================================
# # g = np.arange(-3,3,0.01)
# # test_scatter = []
# # test_norm = []
# # test_ycept = []
# # test_slope = []
# # gamma = []
# # cov = np.cov(sigma_Ysz,sigma_c)
# # for i in g:
# #     yarray = log_Lx - i*log_c
# #     yerr = np.sqrt( (sigma_Lx)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
# #     xarray = log_Ysz_new
# #     xerr = sigma_Ysz
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
# # 
# # =============================================================================
# cov = np.cov(sigma_Ysz,sigma_c)
# 
# yarray = log_Lx - 0.32*log_c
# xarray = log_Ysz_new
# yerr = np.sqrt( (sigma_Lx)**2 + (0.32*sigma_c)**2 + 2*0.32*cov[0][1] )
# xerr = sigma_Ysz 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Ysz,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($Y_{sz}/C^{0.32}$-T)' )
# plt.errorbar(log_Ysz,log_Lx,xerr= sigma_Ysz,yerr = sigma_Lx,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($Y_{SZ}-T$)')
# 
# z = test_Ycept+ test_Slope* log_Ysz_new
# z1 = ycept + Slope* log_Ysz_new
# 
# plt.plot(log_Ysz,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Ysz,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# plt.xlabel('log(Ysz)')
# plt.ylabel('$log(L_{X}/C)$')
# plt.title('$L_{X}$/C - $Y_{SZ}$ scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)gamma_range = np.arange(-4,0,0.1)
# 
# 
# ################################################################################
#                           # To constain gamma an g simultaneously
# ##############################################################################3##
# # =============================================================================
# # gamma_range = np.arange(-4,-3,0.01)
# # g_range = np.arange(0.2,0.4,0.01)
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
# #         Ysz_new = Ysz  /60
# #         log_Ysz = np.log10(Ysz)
# #         log_Ysz_new = np.log10(Ysz_new)
# #         sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# #         
# #         Lx = sz['Lx(1e44)']
# #         Lx_new = Lx * (E)**(i) #### The power of redshift evolution for this relations is -5/3
# #         log_Lx = np.log10(Lx_new)
# #         sigma_Lx = 0.4343*sz['eL(%)']/100
# #         
# #                 
# #         c = sz['c']/np.median(sz['c'])
# #         e_c = sz['e_c']
# #         log_c = np.log10(c)
# #         sigma_c = 0.4343 * e_c/c
# #         
# #         cov = nsp.cov(sigma_Lx,sigma_c)
# # 
# #         yarray = log_Lx - j*log_c
# #         xarray = log_Ysz_new
# #         yerr = np.sqrt( (sigma_Lx)**2 + (j*sigma_c)**2 + 2*j*cov[0][1] )
# #         xerr = sigma_Ysz
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

