#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import time
import seaborn as sns

r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')

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
#for i in range(len(theta)):-
 ##   R_k = (theta[i] * d_A[i])
   # R_kpc.append(R_k.value)

#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
rl['R kpc'] = R_kpc
rl['Rmin kpc'] = Rmin_kpc
rl['Rmax kpc'] = Rmax_kpc

# # To constrain gamma
# =============================================================================
# re_range = np.arange(-3,5,0.01)
# norm = []
# scatter = []
# slope = []
# re = []
# 
# # A for loop t run throught each value of re_range and getting Norm, slope and scatter values.
# for i in re_range:
#     
#     R = rl['R kpc']
#     R_min = rl['Rmin kpc']
#     R_max = rl['Rmax kpc']
#     
#     R_new = (R_kpc/250) * (E**(i))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# 
#     Lx = rl['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*rl['eL']/100
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
#     norm.append(Norm)
#     slope.append(Slope)
#     scatter.append(Scatter)
#     re.append(i)
#     
# # To return the index which corresponds to the minimum scatter    
# p = np.where(scatter == np.min(scatter))
# P = p[0]
# re[P[0]]
# # gamma = -0.68
# =============================================================================

# Uncertainity on gamma
# =============================================================================
# RE = []
# for j in range(1000):
# 
#     random_clusters = rl.sample(n = len(rl), replace = True)
#     re_range = np.arange(-14,14,0.1)
#     scatter = []
#     re = []
#     for i in re_range:
#         random_clusters = rl.sample(n = len(rl), replace = True)
# 
#         Z = (random_clusters['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         R = random_clusters['R kpc']
#         R_min = random_clusters['Rmin kpc']
#         R_max = random_clusters['Rmax kpc']
#         R_new = (R/250) * (E**(i))
#         log_r = np.log10(R_new)
#         sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
#         Lx =random_clusters['Lx']
#         log_Lx= np.log10(Lx)
#         sigma_Lx = 0.4343*random_clusters['eL']/100
# 
#         ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
#         scatter.append(Scatter)
#         re.append(i)
# 
#     ## To return the index which corresponds to the minimum scatter    
#     p = np.where(scatter == np.min(scatter))
#     P = p[0]
#     re[P[0]],scatter[P[0]]
#     RE.append(re[P[0]])
#     print(j)
# general_functions.calculate_asymm_err(RE)
# bestfit_bootstrap_dict = {'Gamma': RE}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_gamma_bootstrap_BCES.csv')
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_gamma_bootstrap_BCES.csv')
# RE = data['Gamma']
# general_functions.calculate_asymm_err(RE)
# =============================================================================

Z = (rl['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R = rl['R kpc']
R_min = rl['Rmin kpc']
R_max = rl['Rmax kpc']
R_new = (R/250) * (E**(-0.68))
log_r = np.log10(R_new)
sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
err_r = [(R - R_min) , (R_max-R) ]


sns.set_context('paper')
weights = np.ones_like(R)/len(R)
plt.hist(R, bins=7, label = 'Normalizations a' )
#plt.axvline(np.median(T), label='median', color = 'black')

plt.xlabel(r'$R$ ($\mathrm{kpc}$)')
plt.ylabel('No. of clusters')
#plt.title('$L_{X}-T$ bootstrap normalizations')
#plt.legend(loc = 'upper right')
plt.xlim(0,1400)
plt.ylim(0,200)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/R_histogram.png',dpi = 300,bbox_inches="tight")
plt.show()



Lx = rl['Lx']
log_Lx= np.log10(Lx)
sigma_Lx = 0.4343*rl['eL']/100
err_Lx = rl['eL']*Lx/100
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)

sns.set_context('paper')
Lx_linspace = np.linspace(0.0001,125,100)
z = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept, Slope)
plt.errorbar(Lx,R,yerr=err_r,xerr=err_Lx,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(Lx)})')
plt.plot(Lx_linspace,z,label='Best fit',color='green')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.title(r'$R-L_{\mathrm{X}}$ best fit')
plt.legend(loc = 'lower right')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Lx_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# =============================================================================
# plt.errorbar(log_Lx,log_r,yerr= sigma_r,xerr=sigma_Lx,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(Lx)})')
# z = ycept * log_Lx*Slope
# plt.plot(log_Lx,z,label='Best fit',color='black')
# plt.show()
# 
# 
# z1 = ycept + Slope*log_Lx
# general_functions.calculate_chi_red(log_r, z1, sigma_r, sigma_Lx, Scatter, Slope)
# 
# plt.errorbar(Z,z1-log_r,yerr=sigma_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_Lx)})')
# #plt.plot(log_Lx,z1,label='Best fit',color='black')
# 
# plt.axhline(0)
# #plt.xlim(-1.5,1.2)
# #plt.ylim(2.5,-2.5)
# plt.xlabel(' $R*E(z)^{-0.68}$ / 250 [kpc]')
# plt.ylabel('Residuals')
# plt.title('$R-L_{X}$ residuals against redshift ')
# #plt.savefig('R-Lx_residualsVz.png',dpi=300)
# plt.show()
# =============================================================================

# ### Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = rl.sample(n = len(rl), replace = True)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(-0.68))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#     Lx =random_clusters['Lx']
#     log_Lx= np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL']/100
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_all_BCES.csv')
# 
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_all_BCES.csv')
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




############################################################

                # Cutting galaxy groups based on mass

###########################################################

rl = general_functions.removing_galaxy_groups(rl)
Z = (rl['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R = rl['R kpc']
R_min = rl['Rmin kpc']
R_max = rl['Rmax kpc']
R_new = (R/250) * (E**(-0.68))
log_r = np.log10(R_new)
sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
err_r = [R - R_min , R_max-R ]


Lx = rl['Lx']
log_Lx= np.log10(Lx)
sigma_Lx = 0.4343*rl['eL']/100
err_Lx = rl['eL']*Lx/100
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)


sns.set_context('paper')
Lx_linspace = np.linspace(0.0001,125,100)
z = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_Mcut, Slope_Mcut)
plt.errorbar(Lx,R,yerr=err_r,xerr=err_Lx,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(Lx)})')
plt.plot(Lx_linspace,z,label='Best fit',color='green')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.legend(loc = 'lower right')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.title(r'$R-L_{\mathrm{X}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-Lx_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()



# ### Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = rl.sample(n = len(rl), replace = True)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(-0.68))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#     Lx =random_clusters['Lx']
#     log_Lx= np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL']/100
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_all(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_all(Mcut)_BCES.csv')
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




############################################################################
 
        # Adding the concentration parameter to the scaling relations

##############################################################################

# =============================================================================
# r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')
# r.rename({'# Name':'Cluster'},axis=1,inplace=True)
# r = general_functions.cleanup(r)
# rl = r[r['R'] > 2]
# R_old = rl['R']
# R_new = general_functions.correct_psf(R_old)
# Rmin_old = rl['R'] - rl['Rmin']
# Rmax_old = rl['R'] + rl['Rmax']
# Rmax_new = general_functions.correct_psf(Rmax_old)
# Rmin_new = general_functions.correct_psf(Rmin_old)
# 
# omega_m = 0.3
# omega_lambda = 0.7
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# Z = (rl['z'])
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
# rl['R kpc'] = R_kpc
# rl['Rmin kpc'] = Rmin_kpc
# rl['Rmax kpc'] = Rmax_kpc
# 
# 
# thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
# thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
# thesis_table = general_functions.cleanup(thesis_table)
# rl = pd.merge(rl, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
# 
# ########## Defining vriables for old scaling relation ############
# omega_m = 0.3
# omega_lambda = 0.7
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# Z = (rl['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# R = rl['R kpc']
# R_min = rl['Rmin kpc']
# R_max = rl['Rmax kpc']
# R_new = (R/250) * (E**(0.02))
# log_r = np.log10(R_new)
# sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# err_r = [(R_new - R_min/250) , (R_max/250-R_new) ]
# 
# 
# Lx = rl['Lx']
# log_Lx= np.log10(Lx)
# sigma_Lx = 0.4343*rl['eL']/100
# err_Lx = rl['eL']*Lx/100
# ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# 
# ############ Adding new parameter for the new scaling relation ############
# 
# c = rl['c']/np.median(rl['c'])
# e_c = rl['e_c']
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
# # =============================================================================
# # g for concentration parameter is -0.44
# 
# yarray = log_r + 0.440*log_c
# xarray = log_Lx
# yerr = np.sqrt( (sigma_r)**2 + (-0.44*sigma_c)**2 + 2*-0.44*cov[0][1] )
# xerr = sigma_Lx 
# test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# 
# plt.errorbar(log_Lx,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($R/C^{-0.44}$-$L_{X}$)' )
# plt.errorbar(log_Lx,log_r,xerr= sigma_Lx,yerr = sigma_r,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($R-L_{X}$)')
# 
# z = test_Ycept+ test_Slope* log_Lx
# z1 = ycept + Slope* log_Lx
# 
# plt.plot(log_Lx,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
# plt.plot(log_Lx,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )
# 
# plt.xlabel('$log_{10}$(Lx / $erg.s^{-1})$')
# plt.ylabel('$log_{10}{Y}$')
# plt.title('$R/c - L_{X}$scaling relation')
# plt.legend(loc='best')
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# ################################################################################
#                           # To constain gamma and g simultaneously
# ##############################################################################3##
# re_range = np.arange(2,3,0.01)
# g_range = np.arange(-1,0.5,0.01)
# 
# test_scatter = []
# test_norm = []
# test_slope = []
# gamma = []
# g = []
# for i in re_range:
#     for j in g_range:
#         Z = (rl['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         R = rl['R kpc']
#         R_min = rl['Rmin kpc']
#         R_max = rl['Rmax kpc']
#         R_new = (R/250) * (E**(i))
#         log_r = np.log10(R_new)
#         sigma_r = 0.4343 * ((R_max-R_min)/(2*R))    
#         Lx = rl['Lx']
#         log_Lx= np.log10(Lx)
#         sigma_Lx = 0.4343*rl['eL']/100
#         
#         c = rl['c']/np.median(rl['c'])
#         e_c = rl['e_c']
#         log_c = np.log10(c)
#         sigma_c = 0.4343 * e_c/c
#         cov = np.cov(sigma_r,sigma_c)
#         
#         yarray = log_r - j*log_c
#         yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#         xarray = log_Lx
#         xerr = sigma_Lx
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
# # Bootstrap for gamma and g
# # =============================================================================
# # start = time.time()
# # gamma_range = np.arange(1.5,3.5,0.01)
# # g_range = np.arange(-0.6,-0.2,0.01)
# # test_scatter = []
# # test_norm = []
# # test_slope = []
# # gamma = []
# # gamma_bootstrap = []
# # g_bootstrap = []
# # g = []
# # cov = np.cov(sigma_r,sigma_c)
# # for k in range(1000):
# #     random_clusters = rl.sample(n=len(rl), replace = True)
# #     for i in gamma_range:
# #         for j in g_range:
# #             Z = (random_clusters['z'])
# #             E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #             R = random_clusters['R kpc']
# #             R_min = random_clusters['Rmin kpc']
# #             R_max = random_clusters['Rmax kpc']
# #             R_new = (R/250) * (E**(i))
# #             log_r = np.log10(R_new)
# #             sigma_r = 0.4343 * ((R_max-R_min)/(2*R))    
# #             Lx = random_clusters['Lx']
# #             log_Lx= np.log10(Lx)
# #             sigma_Lx = 0.4343*random_clusters['eL']/100
# #             
# #             c = random_clusters['c']/np.median(random_clusters['c'])
# #             e_c = random_clusters['e_c']
# #             log_c = np.log10(c)
# #             sigma_c = 0.4343 * e_c/c
# #             cov = np.cov(sigma_r,sigma_c)
# #             
# #             yarray = log_r - j*log_c
# #             yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
# #             xarray = log_Lx
# #             xerr = sigma_Lx
# #             
# #             ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# #             test_scatter.append(Scatter)
# #             test_norm.append(Norm)
# #             test_slope.append(Slope)
# #             gamma.append(i)
# #             g.append(j)
# #     print(k)
# #     p = np.where(test_scatter == np.min(test_scatter))
# #     P = p[0]
# #     gamma_bootstrap.append(gamma[P[0]])
# #     g_bootstrap.append(g[P[0]])
# # bestfit_bootstrap_dict = {'gamma': gamma_bootstrap, 'g': g_bootstrap}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Lx_gamma&_bootstrap.csv')
# # print(time.time()-start.time())
# # =============================================================================
# 
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/R-Lx_gamma&_bootstrap.csv')
# gamma = data['gamma']
# g =  data['g']
# np.std(gamma)
# err_gamma = general_functions.calculate_asymm_err(gamma)
# err_g = general_functions.calculate_asymm_err(g)
# 
# plt.hist(g, bins= 6)
# plt.show()
# =============================================================================
