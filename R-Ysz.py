#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns



r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')

r = general_functions.cleanup(r)
ry = r[r['R'] > 2]
StN = ry['Ysz']/ry['eY']
ry = ry[StN > 2]
R_old = ry['R']
R_new = general_functions.correct_psf(R_old)
Rmin_old = ry['R'] - ry['Rmin']
Rmax_old = ry['R'] + ry['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)
Rmin_new = general_functions.correct_psf(Rmin_old)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (ry['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000

theta = (R_new/60)*np.pi/180
R_kpc = theta * d_A.value
#for i in range(len(theta)):
 ##   R_k = (theta[i] * d_A[i])
   # R_kpc.append(R_k.value)

#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
ry['R kpc'] = R_kpc
ry['Rmin kpc'] = Rmin_kpc
ry['Rmax kpc'] = Rmax_kpc

# ## To constrain gamma
# =============================================================================
# re_range = np.arange(-2,5,0.01)
# norm = []
# scatter = []
# slope = []
# re = []
# for i in re_range:
#     
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (ry['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     d_A = cosmo.angular_diameter_distance(Z)*1000
# 
#         
#     R = ry['R kpc']
#     R_min = ry['Rmin kpc']
#     R_max = ry['Rmax kpc'] 
#     r_new = (R/250) * (E**(i))
#     log_r = np.log10(r_new)
#     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
#     
#     Ysz_arcmin = ry['Ysz']
#     e_Y_arcmin = ry['eY']
#     Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /25
#     log_Ysz = np.log10(Ysz)
#     log_Ysz_new = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
#   
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_Ysz_new, sigma_Ysz, log_r, sigma_r)
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
#gamma = 0.87
# Uncertainties on gamma

# =============================================================================
# RE = []
# for j in range(1000):
# 
#     random_clusters = ry.sample(n = len(ry), replace = True)
#     re_range = np.arange(-10,10,0.1)
#     scatter = []
#     re = []
# 
#     ## A for loop t run throught each value of re_range and getting Norm, slope and scatter values.
#     for i in re_range:
# 
#         omega_m = 0.3
#         omega_lambda = 0.7
#         cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#         Z = (random_clusters['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         d_A = cosmo.angular_diameter_distance(Z)*1000
# 
# 
#         R = random_clusters['R kpc']
#         R_min = random_clusters['Rmin kpc']
#         R_max = random_clusters['Rmax kpc'] 
#         r_new = (R/250) * (E**(i))
#         log_r = np.log10(r_new)
#         sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# 
#         Ysz_arcmin = random_clusters['Ysz']
#         e_Y_arcmin = random_clusters['eY']
#         Ysz = (Ysz_arcmin * (d_A**2) * (np.pi / (60*180))**2)
#         Ysz_new = Ysz /25
#         log_Ysz = np.log10(Ysz)
#         log_Ysz_new = np.log10(Ysz_new)
#         sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
#         #cov = np.cov(sigma_r,sigma_T)
#         ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)
#        
#         scatter.append(Scatter)
#         re.append(i)
# 
#     p = np.where(scatter == np.min(scatter))
#     P = p[0]
#     RE.append(re[P[0]])
#     print(j)
# general_functions.calculate_asymm_err(RE)
# =============================================================================

Z = (ry['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R = ry['R kpc']
R_min = ry['Rmin kpc']
R_max = ry['Rmax kpc'] 
r_new = (R/250) * (E**(0.87))
log_r = np.log10(r_new)
sigma_r = 0.4343 * (R_max - R_min)/(2*R)
err_r = [R-R_min,R_max-R]
Ysz_arcmin = ry['Ysz']
e_Y_arcmin = ry['eY']
Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz /25
log_Ysz = np.log10(Ysz)
log_Ysz_new = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
err_Ysz = ((e_Y_arcmin)* (d_A.value**2) * (np.pi / (60*180))**2)* (E**(-0.170))
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)


sns.set_context('paper')
Ysz_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept, Slope)
plt.plot(Ysz_linspace,z,label='Best fit', color ='green')
plt.errorbar(Ysz,R,yerr=err_r,xerr=err_Ysz,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_r)})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Y_{\mathrm{SZ}} \,(\mathrm{kpc}^{2}$) ')
plt.ylabel(r' $R*E(z)^{0.87}$ (kpc)')
plt.title(r'$R-Y_{\mathrm{SZ}}$ best fit')
plt.legend(loc='lower right')
plt.xlim(0.3,1500)
plt.ylim(15,3000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Ysz_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


# Plotting residuals
# =============================================================================
# plt.errorbar(r_new,z-r_new,yerr=sigma_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red' )
# plt.axhline(0)
# #plt.xlim(-1.5,1.2)
# #plt.ylim(1.9,-1.6)
# plt.ylabel(' Residuals')
# plt.xlabel('$R*E(z)^{-1.08}$ /250 [kpc]')
# plt.title('R-Ysz residuals')
# #plt.savefig('R-Ysz_errors.png',dpi=300)
# plt.legend(bbox_to_anchor=[0.7,0.2])
# #plt.savefig('R-Ysz_residualsVr.png',dpi=300)
# plt.show()
# =============================================================================


# Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = ry.sample(n = len(ry), replace = True)
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     d_A = cosmo.angular_diameter_distance(Z)*1000
# 
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc'] 
#     r_new = (R/250) * (E**(0.87))
#     log_r = np.log10(r_new)
#     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# 
#     Ysz_arcmin = random_clusters['Ysz']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (d_A**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /25
#     log_Ysz = np.log10(Ysz)
#     log_Ysz_new = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Ysz_all_BCES.csv')
# =============================================================================




data = pd.read_csv('//home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_all_BCES.csv')
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


ry = general_functions.removing_galaxy_groups(ry)
Z = (ry['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R = ry['R kpc']
R_min = ry['Rmin kpc']
R_max = ry['Rmax kpc'] 
r_new = (R/250) * (E**(0.87))
log_r = np.log10(r_new)
sigma_r = 0.4343 * (R_max - R_min)/(2*R)
err_r = [R-R_min,R_max-R]

Ysz_arcmin = ry['Ysz']
e_Y_arcmin = ry['eY']
Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new = Ysz /25
log_Ysz = np.log10(Ysz)
log_Ysz_new = np.log10(Ysz_new)
sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
err_Ysz = ((e_Y_arcmin)* (d_A.value**2) * (np.pi / (60*180))**2)
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)




sns.set_context('paper')
Ysz_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept_Mcut, Slope_Mcut)
plt.plot(Ysz_linspace,z,label='Best fit', color ='green')
plt.errorbar(Ysz,R,yerr=err_r,xerr=err_Ysz,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_r)})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Y_{\mathrm{SZ}} \,(\mathrm{kpc}^{2}$) ')
plt.ylabel(r' $R*E(z)^{0.87}$ (kpc)')
plt.title(r'$R-Y_{\mathrm{SZ}}$ best fit $(M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc='lower right')
plt.xlim(0.3,1500)
plt.ylim(15,3000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Ysz_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


# Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     
#     random_clusters = ry.sample(n = len(ry), replace = True)
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     d_A = cosmo.angular_diameter_distance(Z)*1000
# 
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc'] 
#     r_new = (R/250) * (E**(0.87))
#     log_r = np.log10(r_new)
#     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# 
#     Ysz_arcmin = random_clusters['Ysz']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (d_A**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /25
#     log_Ysz = np.log10(Ysz)
#     log_Ysz_new = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)
# 
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Ysz_all(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('//home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_all(Mcut)_BCES.csv')
norm= data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut =  general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut =  general_functions.calculate_asymm_err(scatter)


print('The best fit parameters are :')
print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')

# =============================================================================
# weights = np.ones_like(RE)/len(RE)
# plt.hist(RE,bins=20, weights = weights)
# plt.xlabel('$\gamma_{R-Y_{SZ}}$')
# plt.ylabel('Count')
# plt.title('Bootstrap for $\gamma_{R-Y_{SZ}}$')
# np.max(RE),np.min(RE)
# general_functions.calculate_asymm_err(RE)
# 
# =============================================================================

# =============================================================================
# r = pd.read_fwf('/home/schubham/Desktop/Half-radii-final.txt',sep ='\\s+')
# r.rename({'# Name':'Cluster'},axis=1,inplace=True)
# r = general_functions.cleanup(r)
# ry = r[r['R'] > 2]
# StN = ry['Ysz']/ry['eY']
# ry = ry[StN >= 4.5]
# R_old = ry['R']
# R_new = general_functions.correct_psf(R_old)
# Rmin_old = ry['R'] - ry['Rmin']
# Rmax_old = ry['R'] + ry['Rmax']
# Rmax_new = general_functions.correct_psf(Rmax_old)
# Rmin_new = general_functions.correct_psf(Rmin_old)
# 
# Z = (ry['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# d_A = cosmo.angular_diameter_distance(Z)*1000
# 
# theta = (R_new/60)*np.pi/180
# R_kpc = theta * d_A.value
# 
# theta_min = (Rmin_new/60)*np.pi/180
# Rmin_kpc = theta_min * d_A.value
# 
# theta_max = (Rmax_new/60)*np.pi/180
# Rmax_kpc = theta_max * d_A.value
# sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
# ry['R kpc'] = R_kpc
# ry['Rmin kpc'] = Rmin_kpc
# ry['Rmax kpc'] = Rmax_kpc
# 
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# ry = pd.merge(ry, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# 
# Z = (ry['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# d_A = cosmo.angular_diameter_distance(Z)*1000
# 
# R = ry['R kpc']
# R_min = ry['Rmin kpc']
# R_max = ry['Rmax kpc'] 
# r_new = (R/250) * (E**(0.87))
# log_r = np.log10(r_new)
# sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# err_r = [(R-R_min)/250,(R_max-R)/250]
# 
# Ysz_arcmin = ry['Ysz']
# e_Y_arcmin = ry['eY']
# Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
# Ysz_new = Ysz /25
# log_Ysz = np.log10(Ysz)
# log_Ysz_new = np.log10(Ysz_new)
# sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# err_Ysz = ((e_Y_arcmin)* (d_A.value**2) * (np.pi / (60*180))**2)
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Ysz_new,sigma_Ysz,log_r,sigma_r)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = ry['c']/np.median(ry['c'])
# e_c = ry['e_c']
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
# #     xarray = log_Ysz_new
# #     xerr = sigma_Ysz
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
# yarray = log_r - (-0.380)*log_c
# xarray = log_Ysz_new
# yerr = np.sqrt( (sigma_r)**2 + (-0.38*sigma_c)**2 + 2*-0.38*cov[0][1] )
# xerr = sigma_Ysz 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Ysz,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($R/C^{-0.38}-Y_{SZ}$)' )
# plt.errorbar(log_Ysz,log_r,xerr= sigma_Ysz,yerr = sigma_r,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($R-Y_{SZ}$)')
# 
# z = test_Ycept+ test_Slope* log_Ysz_new
# z1 = ycept + Slope* log_Ysz_new
# 
# plt.plot(log_Ysz,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Ysz,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('log(Ysz)')
# plt.ylabel('$log{r}$')
# plt.title('$R/c - Y_{SZ}$scaling relation')
# plt.legend(loc='upper left')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# ################################################################################
#                           # To constain gamma an g simultaneously
# ##############################################################################3##
# re_range = np.arange(1.6,1.80,0.01)
# g_range = np.arange(-0.40,-0.36,0.001)
# 
# test_scatter = []
# test_norm = []
# test_slope = []
# gamma = []
# g = []
# for i in re_range:
#     for j in g_range:
#         Z = (ry['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         d_A = cosmo.angular_diameter_distance(Z)*1000
#         
#         R = ry['R kpc']
#         R_min = ry['Rmin kpc']
#         R_max = ry['Rmax kpc'] 
#         r_new = (R/250) * (E**(i))
#         log_r = np.log10(r_new)
#         sigma_r = 0.4343 * (R_max - R_min)/(2*R)
#         
#         Ysz_arcmin = ry['Ysz']
#         e_Y_arcmin = ry['eY']
#         Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
#         Ysz_new = Ysz /25
#         log_Ysz = np.log10(Ysz)
#         log_Ysz_new = np.log10(Ysz_new)
#         sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
#         
#         c = ry['c']/np.median(ry['c'])
#         e_c = ry['e_c']
#         log_c = np.log10(c)
#         sigma_c = 0.4343 * e_c/c
#         cov = np.cov(sigma_r,sigma_c)
#         yarray = log_r - j*log_c
#         yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#         xarray = log_Ysz_new
#         xerr = sigma_Ysz
#         
#         ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#         test_scatter.append(Scatter)
#         test_norm.append(Norm)
#         test_slope.append(Slope)
#         gamma.append(i)
#         g.append(j)
# 
# p = np.where(test_scatter == np.min(test_scatter))
# P = p[0]
# test_scatter[P[0]],gamma[P[0]],g[P[0]]
# 
# #Uncertainties on gamma and g
# 
# # =============================================================================
# # re_range = np.arange(0,4.5,0.1)
# # g_range = np.arange(-0.6,-0.2,0.01)
# # 
# # test_scatter = []
# # gamma_bootstrap = []
# # g_bootstrap = []
# # gamma = []
# # g = []
# # for k in range(1000):
# #     random_clusters = ry.sample(n = len(ry), replace = True)
# #     
# #     for i in re_range:
# #         for j in g_range:
# #             Z = (random_clusters['z'])
# #             E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #             d_A = cosmo.angular_diameter_distance(Z)*1000
# #             
# #             R = random_clusters['R kpc']
# #             R_min = random_clusters['Rmin kpc']
# #             R_max = random_clusters['Rmax kpc'] 
# #             r_new = (R/250) * (E**(i))
# #             log_r = np.log10(r_new)
# #             sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# #             
# #             Ysz_arcmin = random_clusters['Ysz']
# #             e_Y_arcmin = random_clusters['eY']
# #             Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
# #             Ysz_new = Ysz /25
# #             log_Ysz = np.log10(Ysz)
# #             log_Ysz_new = np.log10(Ysz_new)
# #             sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# #             
# #             c = random_clusters['c']/np.median(random_clusters['c'])
# #             e_c = random_clusters['e_c']
# #             log_c = np.log10(c)
# #             sigma_c = 0.4343 * e_c/c
# #             cov = np.cov(sigma_r,sigma_c)
# #             yarray = log_r - j*log_c
# #             yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
# #             xarray = log_Ysz_new
# #             xerr = sigma_Ysz
# #             
# #             ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #             test_scatter.append(Scatter)
# #            
# #             gamma.append(i)
# #             g.append(j)
# #     
# #     p = np.where(test_scatter == np.min(test_scatter))
# #     P = p[0]
# #     gamma_bootstrap.append(gamma[P[0]])
# #     g_bootstrap.append(g[P[0]])
# #     print(k,gamma[P[0]],g[P[0]])
# #             
# # 
# # bestfit_bootstrap_dict = {'gamma': gamma_bootstrap, 'g': g_bootstrap}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_gamma_g_bootstrap.csv')
# # plt.hist(gamma_bootstrap, bins =8 )
# # plt.show()
# # plt.hist(g_bootstrap, bins=10)
# # plt.xlim(-0.40, -0.37)
# # plt.show()
# # np.max(gamma_bootstrap), np.min(gamma_bootstrap)
# # 
# # 
# # 
# # =============================================================================
# 
# =============================================================================
