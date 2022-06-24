#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM 
import general_functions
import seaborn as sns

#Importing the best fit values of the scaling relation fit

bestfit_values = pd.read_csv('/home/schubham/Thesis/Thesis/Data/best_fit_parameters.csv')
bestfit_Norm = bestfit_values['Norm_all'][1]
err_bestfit_Norm = bestfit_values['err_Norm_all'][1]
bestfit_Slope = bestfit_values['Slope_all'][1]
err_bestfit_Slope = bestfit_values['err_Slope_all'][1]
bestfit_Scatter = bestfit_values['Scatter_all'][1]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][1]



sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')
sz = sz.rename({'#Cluster':'Cluster'},axis=1)
sz = general_functions.cleanup(sz)
StN = sz['Y(r/no_ksz,arcmin^2)']/sz['e_Y']
sz = sz[StN >2]

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv',sep =',')
thesis_table = general_functions.cleanup(thesis_table)
sz_data = pd.merge(sz, thesis_table, left_on = sz['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')

g = sz_data.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

# Best fit params
omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z_c = (CC_clusters['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z_c) * 1000
Ysz_arcmin = CC_clusters['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = CC_clusters['e_Y']
Ysz_c = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c * E_c/35
log_Ysz_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin/Ysz_arcmin

T_c = CC_clusters['T(keV)']
T_new_c = T_c/4.5
log_T_c = np.log10(T_c)
log_T_new_c = np.log10(T_new_c)
sigma_T_c = 0.4343*((CC_clusters['Tmax']-CC_clusters['Tmin'])/(2*T_c))
err_Ysz_c = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E_c
err_T_c = [T_c - CC_clusters['Tmin'], CC_clusters['Tmax']-T_c]
ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Ysz_c,sigma_Ysz_c)


# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
# #     
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# # 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# # 
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
# #     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# # 
# #     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# # 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_CC_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c =  general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)


# NCC clusters
Z_n = (NCC_clusters['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z_n) * 1000
Ysz_arcmin = NCC_clusters['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = NCC_clusters['e_Y']
Ysz_n = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n * E_n/35
log_Ysz_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin/Ysz_arcmin

T_n = NCC_clusters['T(keV)']
T_new_n = T_n/4.5
log_T_n = np.log10(T_n)
log_T_new_n = np.log10(T_new_n)
sigma_T_n = 0.4343*((NCC_clusters['Tmax'] - NCC_clusters['Tmin'])/(2*T_n))
err_Ysz_n = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E_n
err_T_n = [NCC_clusters['Tmin']-T_n, NCC_clusters['Tmax']-T_n]

ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_T_new_n,sigma_T_n,log_Ysz_n,sigma_Ysz_n)



# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
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
#     
#     ycept,Norm,Slope,Scatter =general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
#     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_NCC_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_NCC_BCES.csv')
norm_n = data['Normalization']
slope_n = data['Slope']
scatter_n = data['Scatter']

errnorm_n = general_functions.calculate_asymm_err(norm_n)
errslope_n = general_functions.calculate_asymm_err(slope_n)
errscatter_n =  general_functions.calculate_asymm_err(scatter_n)



print('CC best fits:')
print(f'Normalization : {np.round(Norm_c,3)} +/- {np.round(errnorm_c,3)}')
print(f'Slope : {np.round(Slope_c,3)} +/- {np.round(errslope_c,3)}')
print(f'Scatter: {np.round(Scatter_c,3)} +/- {np.round(errscatter_c,3)}')

print('NCC best fits:')

print(f'Normalization : {np.round(Norm_n,3)} +/- {np.round(errnorm_n,3)}')
print(f'Slope : {np.round(Slope_n,3)} +/- {np.round(errslope_n,3)}')
print(f'Scatter: {np.round(Scatter_n,3)} +/- {np.round(errscatter_n,3)}')


sns.set_context('paper')
T_linspace = np.linspace(0.5,100,100)

z_c = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept_n, Slope_n)

# =============================================================================
# 
# z_c = Norm_c * T_new_c ** Slope_c
# z_n = Norm_n * T_new_n ** Slope_n
# =============================================================================
plt.plot(T_linspace,z_c, label = 'Best fit ',color = 'black')
plt.plot(T_linspace,z_n, label = 'Best fit NCC ',color = 'blue')
plt.errorbar(T_c,Ysz_c*E_c,xerr=err_T_c, yerr=err_Ysz_c,color = 'green',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'CC clusters ({len(T_c)})' )
plt.errorbar(T_n,Ysz_n*E_n,xerr=err_T_n, yerr=err_Ysz_n,color = 'red',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'NCC clusters ({len(T_n)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.xlabel('T (keV)')
plt.ylabel(r'$Y_{\mathrm{SZ}}*E(z)$ ($\mathrm{kpc}^{2}$)')
plt.title(r'$Y_{\mathrm{SZ}}-T$ best fit ')

plt.legend(loc='lower right')
plt.xlim(0.85, 25)
plt.ylim(0.2,1500)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Ysz-T_ccVncc_bestfit.png',dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# print('The best fit parameters for CC are :')
# print(f'Normalization : {np.round(Norm_c,3)} +/- {np.round(errnorm_c,3)}')
# print(f'Slope : {np.round(Slope_c,3)} +/- {np.round(errslope_c,3)}')
# print(f'Scatter: {np.round(Scatter_c,3)} +/- {np.round(errscatter_c,3)}')
# 
# print('The best fit parameters for NCC are :')
# print(f'Normalization : {np.round(Norm_n,3)} +/- {np.round(errnorm_n,3)}')
# print(f'Slope : {np.round(Slope_n,3)} +/- {np.round(errslope_n,3)}')
# print(f'Scatter: {np.round(Scatter_n,3)} +/- {np.round(errscatter_n,3)}')
# =============================================================================

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')

print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))




####################################################################3
         # Removing galaxy groups based on mass cut
####################################################################3



bestfit_Norm_clusters = bestfit_values['Norm_clusters'][1]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][1]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][1]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][1]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][1]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][1]

## Cutting groups based on Mass ######
sz_data = general_functions.removing_galaxy_groups(sz_data)

g = sz_data.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

# Best fit params
omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z_c = (CC_clusters['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z_c) * 1000
Ysz_arcmin = CC_clusters['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = CC_clusters['e_Y']
Ysz_c = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c * E_c/35
log_Ysz_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin/Ysz_arcmin

T_c = CC_clusters['T(keV)']
T_new_c = T_c/4.5
log_T_c = np.log10(T_c)
log_T_new_c = np.log10(T_new_c)
sigma_T_c = 0.4343*((CC_clusters['Tmax']-CC_clusters['Tmin'])/(2*T_c))
err_Ysz_c = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E_c
err_T_c = [T_c - CC_clusters['Tmin'], CC_clusters['Tmax']-T_c]
ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Ysz_c,sigma_Ysz_c)


# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
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
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
#     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_CC(M_cut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_CC(M_cut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut =  general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)


# NCC clusters
Z_n = (NCC_clusters['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z_n) * 1000
Ysz_arcmin = NCC_clusters['Y(r/no_ksz,arcmin^2)']
e_Y_arcmin = NCC_clusters['e_Y']
Ysz_n = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n * E_n/35
log_Ysz_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin/Ysz_arcmin

T_n = NCC_clusters['T(keV)']
T_new_n = T_n/4.5
log_T_n = np.log10(T_n)
log_T_new_n = np.log10(T_new_n)
sigma_T_n = 0.4343*((NCC_clusters['Tmax'] - NCC_clusters['Tmin'])/(2*T_n))
err_Ysz_n = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E_n
err_T_n = [NCC_clusters['Tmin']-T_n, NCC_clusters['Tmax']-T_n]

ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_T_new_n,sigma_T_n,log_Ysz_n,sigma_Ysz_n)



# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
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
#     
#     ycept,Norm,Slope,Scatter =general_functions.calculate_bestfit(log_T_new,sigma_T,log_Ysz,sigma_Ysz)
# 
#     
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-T_NCC(M_cut)_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-T_NCC(M_cut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut = general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)



# =============================================================================
# z_c = Norm_c_Mcut * T_new_c ** Slope_c_Mcut
# z_n = Norm_n_Mcut * T_new_n ** Slope_n_Mcut
# sns.set_context('paper')
# plt.plot(T_c,z_c, label = 'Best fit ',color = 'black')
# plt.plot(T_n,z_n, label = 'Best fit NCC ',color = 'blue')
# plt.errorbar(T_c,Ysz_new_c,xerr=err_T_c, yerr=err_Ysz_c,color = 'green',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'CC clusters ({len(T_c)})' )
# plt.errorbar(T_n,Ysz_new_n,xerr=err_T_n, yerr=err_Ysz_n,color = 'red',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'NCC clusters ({len(T_n)})' )
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('T (keV)')
# plt.ylabel(r'$Y_{\mathrm{SZ}}*E(z)$ ($35^{-1} \,\mathrm{kpc}^{2}$)')
# plt.title(r'$Y_{\mathrm{SZ}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}M_{\odot}$)')
# plt.legend(loc='lower right')
# plt.xlim(0.85, 25)
# plt.ylim(0.003,50)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Ysz-T_ccVncc_Mcut_bestfit.png',dpi=300, bbox_inches="tight")
# plt.show()
# =============================================================================


sns.set_context('paper')
T_linspace = np.linspace(0.5,100,100)

z_c = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 35, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(T_linspace,z_c, label = 'Best fit ',color = 'black')
plt.plot(T_linspace,z_n, label = 'Best fit NCC ',color = 'blue')
plt.errorbar(T_c,Ysz_c*E_c,xerr=err_T_c, yerr=err_Ysz_c,color = 'green',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'CC clusters ({len(T_c)})' )
plt.errorbar(T_n,Ysz_n*E_n,xerr=err_T_n, yerr=err_Ysz_n,color = 'red',ls='',fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label = f'NCC clusters ({len(T_n)})' )
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right')
plt.xlabel('T (keV)')
plt.ylabel(r'$Y_{\mathrm{SZ}}*E(z)$ ($\mathrm{kpc}^{2}$)')
plt.title(r'$Y_{\mathrm{SZ}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}M_{\odot}$)')

plt.legend(loc='lower right')
plt.xlim(0.85, 25)
plt.ylim(0.2,1500)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Ysz-T_ccVncc_Mcut_bestfit.png',dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# print('The best fit parameters for CC are :')
# print(f'Normalization : {np.round(Norm_c,3)} +/- {np.round(errnorm_c,3)}')
# print(f'Slope : {np.round(Slope_c,3)} +/- {np.round(errslope_c,3)}')
# print(f'Scatter: {np.round(Scatter_c,3)} +/- {np.round(errscatter_c,3)}')
# 
# print('The best fit parameters for NCC are :')
# print(f'Normalization : {np.round(Norm_n,3)} +/- {np.round(errnorm_n,3)}')
# print(f'Slope : {np.round(Slope_n,3)} +/- {np.round(errslope_n,3)}')
# print(f'Scatter: {np.round(Scatter_n,3)} +/- {np.round(errscatter_n,3)}')
# =============================================================================

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut)}')

print(general_functions.percent_diff(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut,bestfit_Norm_clusters, err_bestfit_Norm_clusters))
print(general_functions.percent_diff(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut,bestfit_Slope_clusters, err_bestfit_Slope_clusters))
print(general_functions.percent_diff(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut,bestfit_Scatter_clusters, err_bestfit_Scatter_clusters))


sns.set_context('paper')
fig, ax_plot = plt.subplots()
#ax.scatter(slope_c,norm_c)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=1,label=r'CC (clusters+groups)', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_c,Norm_c,color = 'green')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=1,label=r'NCC (clusters+groups)', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_n,Norm_n,color = 'darkorange')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=1,label=r'CC (clusters)', edgecolor='blue', lw = 1)
general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=3, edgecolor='blue', lw = 1)
plt.scatter(Slope_c_Mcut,Norm_c_Mcut,color = 'blue')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=1,label=r'NCC (clusters)', edgecolor='red', lw = 1)
general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=3, edgecolor='red', lw = 1)
plt.scatter(Slope_n_Mcut,Norm_n_Mcut,color = 'red')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
    
plt.xlim(1.75,3.4)
plt.ylim(0.3,1.2)
plt.legend(loc= 'lower right', prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title(r'$Y_{\mathrm{SZ}}-T$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Ysz-T_ccVncc_contours.png', dpi = 300, bbox_inches="tight")

plt.show()

# =============================================================================
# C_c = CC_clusters['c']
# err_c_c = CC_clusters['e_c']
# z_c = ycept_c + Slope_c* log_T_new_c
# z_n = ycept_n + Slope_n* log_T_new_n
# 
# C_n = NCC_clusters['c']
# err_c_n = NCC_clusters['e_c']
# plt.errorbar(C_c,z_c-log_Ysz_c, yerr = sigma_Ysz_c, xerr= err_c_c ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(z_c)})')
# plt.errorbar(C_n,z_n-log_Ysz_n, yerr = sigma_Ysz_n, xerr= err_c_n ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(z_n)})')
# plt.ylim(-1,1)
# plt.xlim(-0.1, 0.8)
# plt.xlabel('Concentration')
# plt.ylabel('$\Delta log_{10}Y_{SZ}$')
# plt.title('$Y_{SZ}-T$ residuals')
# plt.legend(loc = 'best')
# plt.axhline(0, color = 'black')
# plt.axvline(0.18, color= 'blue', ls= '--', label='Threshold')
# #plt.savefig('Y-c_residuals.png',dpi=300)
# plt.show()
# 
# 
# 
# =============================================================================

