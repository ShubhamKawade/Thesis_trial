#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns


#Importing the best fit values of the scaling relation fit
bestfit_values = pd.read_csv('/home/schubham/Thesis/Thesis/Data/best_fit_parameters.csv')
bestfit_Norm = bestfit_values['Norm_all'][9]
err_bestfit_Norm = bestfit_values['err_Norm_all'][9]
bestfit_Slope = bestfit_values['Slope_all'][9]
err_bestfit_Slope = bestfit_values['err_Slope_all'][9]
bestfit_Scatter = bestfit_values['Scatter_all'][9]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][9]


r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half-radii-T-NEW-2_eeHIF_mass.csv')
r.rename({'#Name':'Cluster'},axis=1,inplace=True)
r = general_functions.cleanup(r)
rt = r[r['R'] > 2]
R_old = rt['R']
R_new = general_functions.correct_psf(R_old)
Rmin_old = rt['R'] - rt['Rmin']
Rmax_old = rt['R'] + rt['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)
Rmin_new = general_functions.correct_psf(Rmin_old)

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z = (rt['z'])
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
rt['R kpc'] = R_kpc
rt['Rmin kpc'] = Rmin_kpc
rt['Rmax kpc'] = Rmax_kpc




thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rt_all = pd.merge(rt, thesis_table,left_on = rt['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')
g = rt_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

# For CC clusters
omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

Z_c = (CC_clusters['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
R_c = CC_clusters['R kpc']
R_min_c = CC_clusters['Rmin kpc']
R_max_c = CC_clusters['Rmax kpc']
R_new_c = (R_c/250) * (E_c**(1.89))
log_r_c = np.log10(R_new_c)
err_r_c = [R_c-R_min_c,R_max_c-R_c]
sigma_r_c = 0.4343 * (R_max_c-R_min_c)/(2*R_c)

T_c = CC_clusters['T']
T_new_c = T_c/4.5
log_T_c = np.log10(T_new_c)
log_T_new_c = np.log10(T_c/4.5)
sigma_T_c = 0.4343 * (CC_clusters['Tmax']-CC_clusters['Tmin'])/(2*T_c)
err_T_c = [T_c-CC_clusters['Tmin'], CC_clusters['Tmax']-T_c]

ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_r_c,sigma_r_c)
# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
# 
# 
#     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# 
# 
#     T = random_clusters['T']
#     log_T = np.log(T)
#     log_T_new = np.log10(T/4.5)
#     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_CC_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)


# NCC clusters
Z_n = (NCC_clusters['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5

R_n = NCC_clusters['R kpc']
R_min_n = NCC_clusters['Rmin kpc']
R_max_n = NCC_clusters['Rmax kpc']
R_new_n = (R_n/250) * (E_n**(1.89))
log_r_n = np.log10(R_new_n)
err_r_n = [R_n-R_min_n,R_max_n-R_n]
sigma_r_n = 0.4343 * (R_max_n-R_min_n)/(2*R_n)


T_n = NCC_clusters['T']
T_new_n = T_n/4.5
log_T_n = np.log10(T_n)
log_T_new_n = np.log10(T_new_n)
sigma_T_n = 0.4343 * (NCC_clusters['Tmax']-NCC_clusters['Tmin'])/(2*T_n)
err_T_n = [T_n-NCC_clusters['Tmin'], NCC_clusters['Tmax']-T_n]

ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_T_new_n,sigma_T_n,log_r_n,sigma_r_n)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# 
#     T = random_clusters['T']
#     log_T = np.log(T)
#     log_T_new = np.log10(T/4.5)
#     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_NCC_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_NCC_BCES.csv')
norm_n = data['Normalization']
slope_n = data['Slope']
scatter_n = data['Scatter']

errnorm_n =  general_functions.calculate_asymm_err(norm_n)
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


plt.errorbar(T_c,R_c,yerr=err_r_c,xerr=err_T_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_T_c)})')
plt.errorbar(T_n,R_n,yerr=err_r_n,xerr=err_T_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_T_n)})')
sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept_n, Slope_n)

plt.plot(T_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(T_linspace,z_n,label='Best fit NCC', color ='black')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.3,50)
plt.ylim(10,3000)

#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(' T (keV)')
plt.ylabel(' $R*E(z)^{1.89}$ (kpc)')
plt.title('$R-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-T_ccVncc_bestfit.png',dpi=300,bbox_inches='tight')
plt.show()

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')

print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))




####################################################################3
         # Removing galaxy groups based on mass cut
####################################################################3



bestfit_Norm_clusters = bestfit_values['Norm_clusters'][9]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][9]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][9]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][9]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][9]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][9]


#CC clusters
c_new = general_functions.removing_galaxy_groups(CC_clusters)
Z_c = (c_new['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
R_c = c_new['R kpc']
R_min_c = c_new['Rmin kpc']
R_max_c = c_new['Rmax kpc']
R_new_c = (R_c/250) * (E_c**(1.89))
log_r_c = np.log10(R_new_c)
err_r_c = [(R_c-R_min_c)/250,(R_max_c-R_c)/250]
sigma_r_c = 0.4343 * (R_max_c-R_min_c)/(2*R_c)

T_c = c_new['T']
T_new_c = T_c/4.5
log_T_c = np.log10(T_new_c)
log_T_new_c = np.log10(T_c/4.5)
sigma_T_c = 0.4343 * (c_new['Tmax']-c_new['Tmin'])/(2*T_c)
err_T_c = [T_c-c_new['Tmin'], c_new['Tmax']-T_c]

ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_r_c,sigma_r_c)
# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
# 
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
# 
# 
#     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# 
# 
#     T = random_clusters['T']
#     log_T = np.log(T)
#     log_T_new = np.log10(T/4.5)
#     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_CC(M_cut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_CC(M_cut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut= general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)


# NCC clusters
n_new = general_functions.removing_galaxy_groups(NCC_clusters)
Z_n = (n_new['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5

R_n = n_new['R kpc']
R_min_n = n_new['Rmin kpc']
R_max_n = n_new['Rmax kpc']
R_new_n = (R_n/250) * (E_n**(1.89))
log_r_n = np.log10(R_new_n)
err_r_n = [R_n-R_min_n,R_max_n-R_n]
sigma_r_n = 0.4343 * (R_max_n-R_min_n)/(2*R_n)


T_n = n_new['T']
T_new_n = T_n/4.5
log_T_n = np.log10(T_n)
log_T_new_n = np.log10(T_new_n)
sigma_T_n = 0.4343 * (n_new['Tmax']-n_new['Tmin'])/(2*T_n)
err_T_n = [T_n-n_new['Tmin'], n_new['Tmax']-T_n]

ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_T_new_n,sigma_T_n,log_r_n,sigma_r_n)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
# 
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# 
#     T = random_clusters['T']
#     log_T = np.log(T)
#     log_T_new = np.log10(T/4.5)
#     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_NCC(M_cut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_NCC(M_cut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut =  general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)


plt.errorbar(T_c,R_c,yerr=err_r_c,xerr=err_T_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_T_c)})')
plt.errorbar(T_n,R_n,yerr=err_r_n,xerr=err_T_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_T_n)})')
sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(T_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(T_linspace,z_n,label='Best fit NCC', color ='black')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.3,50)
plt.ylim(10,3000)

#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(' T (keV)')
plt.ylabel(' $R*E(z)^{1.89}$ (kpc)')
plt.title('$R-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-T_ccVncc_bestfit_Mcut.png',dpi=300,bbox_inches='tight')
plt.show()

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut)}')

print(general_functions.percent_diff(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut,bestfit_Norm_clusters, err_bestfit_Norm_clusters))
print(general_functions.percent_diff(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut,bestfit_Slope_clusters, err_bestfit_Slope_clusters))
print(general_functions.percent_diff(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut,bestfit_Scatter_clusters, err_bestfit_Scatter_clusters))




# Plotting uncertainty contours 
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
    
plt.xlim(-0.3,1.0)
plt.ylim(0.4,1.8)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-T$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-T_ccVncc_contours.png' ,dpi=300, bbox_inches="tight")

plt.show()


# =============================================================================
# # 2keV cut
# # CC clusters
# c_new = CC_clusters[CC_clusters['T'] > 2]
# Z_c = (c_new['z'])
# E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
# 
# 
# 
# R_c = c_new['R kpc']
# R_min_c = c_new['Rmin kpc']
# R_max_c = c_new['Rmax kpc']
# R_new_c = (R_c/250) * (E_c**(1.9))
# log_r_c = np.log10(R_new_c)
# err_r_c = [(R_c-R_min_c)/250,(R_max_c-R_c)/250]
# sigma_r_c = 0.4343 * (R_max_c-R_min_c)/(2*R_c)
# 
# 
# T_c = c_new['T']
# T_new_c = T_c/4.5
# log_T_c = np.log10(T_c)
# log_T_new_c = np.log10(T_new_c)
# sigma_T_c = 0.4343 * (c_new['Tmax']-c_new['Tmin'])/(2*T_c)
# err_T_c = [T_c-c_new['Tmin'], c_new['Tmax']-T_c]
# 
# ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_r_c,sigma_r_c)
# 
# # Bootstrap
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # 
# # for j in range(0,10000):
# #     
# #     random_clusters = c_new.sample(n = len(c_new), replace = True)
# # 
# #     omega_m = 0.3
# #     omega_lambda = 0.7
# #     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# # 
# # 
# # 
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(1.9))
# #     log_r = np.log10(R_new)
# # 
# # 
# #     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# # 
# # 
# #     T = random_clusters['T']
# #     log_T = np.log(T)
# #     log_T_new = np.log10(T/4.5)
# #     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
# #     
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-T_CC(2kev)_BCES.csv')
# # 
# # =============================================================================
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_CC(2kev)_BCES.csv')
# norm_c = data['Normalization']
# slope_c = data['Slope']
# scatter_c = data['Scatter']
# 
# errnorm_c = general_functions.calculate_asymm_err(norm_c)
# errslope_c = general_functions.calculate_asymm_err(slope_c)
# errscatter_c =  general_functions.calculate_asymm_err(scatter_c)
# 
# 
# # NCC cluster
# n_new = NCC_clusters[NCC_clusters['T'] > 2]
# 
# omega_m = 0.3
# omega_lambda = 0.7
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
# Z_n = (n_new['z'])
# E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
# R_n = n_new['R kpc']
# R_min_n = n_new['Rmin kpc']
# R_max_n = n_new['Rmax kpc']
# R_new_n = (R_n/250) * (E_n**(1.9))
# log_r_n = np.log10(R_new_n)
# err_r_n = [(R_n-R_min_n)/250,(R_max_n-R_n)/250]
# sigma_r_n = 0.4343 * (R_max_n-R_min_n)/(2*R_n)
# 
# 
# T_n = n_new['T']
# T_new_n = T_n/4.5
# log_T_n = np.log10(T_n)
# log_T_new_n = np.log10(T_new_n)
# sigma_T_n = 0.4343 * (n_new['Tmax']-n_new['Tmin'])/(2*T_n)
# err_T_n = [T_n-n_new['Tmin'], n_new['Tmax']-T_n]
# 
# ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_T_new_n,sigma_T_n,log_r_n,sigma_r_n)
# 
# # Bootstrap
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # 
# # for j in range(0,10000):
# #     
# #     random_clusters = n_new.sample(n = len(n_new), replace = True)
# # 
# # 
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# # 
# # 
# # 
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(1.9))
# #     log_r = np.log10(R_new)
# #     sigma_r = 0.4343 * (R_max-R_min)/(2*R)
# # 
# #     T = random_clusters['T']
# #     log_T = np.log(T)
# #     log_T_new = np.log10(T/4.5)
# #     sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
# #     
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-T_NCC(2keV)_BCES.csv')
# # 
# # =============================================================================
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_NCC(2keV)_BCES.csv')
# norm_n = data['Normalization']
# slope_n = data['Slope']
# scatter_n = data['Scatter']
# 
# errnorm_n =  general_functions.calculate_asymm_err(norm_n)
# errslope_n = general_functions.calculate_asymm_err(slope_n)
# errscatter_n =  general_functions.calculate_asymm_err(scatter_n)
# 
# plt.errorbar(log_T_c,log_r_c,yerr=sigma_r_c,xerr=sigma_T_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
# plt.errorbar(log_T_n,log_r_n,yerr=sigma_r_n,xerr=sigma_T_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_r_n)})')
# 
# #z = best_a + best_b*log_Lx
# z_c = ycept_c + log_T_new_c*Slope_c
# z_n = ycept_n + log_T_new_n*Slope_n
# 
# #plt.axvline(np.log10(0.15),color='Black')
# 
# plt.plot(log_T_c,z_c,label='Best fit CC',color='black')
# plt.plot(log_T_n,z_n,label='Best fit NCC',color='blue')
# 
# #plt.xlim(1.5,25)
# #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# plt.xlabel('$log_{10}(T/4.5$ [keV])')
# plt.ylabel(' $log_{10}(R*E(z)^{1.9}$ / 250 [kpc])')
# plt.title('$R-T$ best fit  ')
# plt.legend(loc='best')
# #plt.savefig('R-Lx_best_fit-CCvNCC().png',dpi=300)
# plt.show()
# 
# 
# 
# 
# plt.errorbar(T_c,R_new_c,yerr=err_r_c,xerr=err_T_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'CC Clusters ({len(log_T_c)})')
# plt.errorbar(T_n,R_new_n,yerr=err_r_n,xerr=err_T_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'NCC Clusters ({len(log_T_n)})')
# 
# z_c = Norm_c * T_new_c**Slope_c
# z_n = Norm_n *T_new_n**Slope_n
# plt.plot(T_c,z_c,label='Best fit CC',color='black')
# plt.plot(T_n,z_n,label='Best fit NCC',color='blue')
# plt.xscale('log')
# plt.yscale('log')
# #plt.xlim(-0.05,1.4)
# #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# plt.xlabel(' T [keV]')
# plt.ylabel(' $R*E(z)^{1.9} $/250 [kpc]')
# plt.title('$R-T$ best fit ')
# plt.legend(bbox_to_anchor=[0.65,0.32])
# #plt.savefig('R-T_best_fit-CCvNCC.png',dpi=300)
# plt.show()
# 
# print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
# print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
# print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')
# 
# 
# 
# 
# print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n))
# print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n))
# print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n))
# 
# =============================================================================
##Residuals against Concentration

# =============================================================================
# bins_c, cdf_c = general_functions.calculate_cdf(T_c, 20)
# bins_n, cdf_n = general_functions.calculate_cdf(T_n, 20)
# plt.plot(bins_c[1:], cdf_c,label = f'CC ({len(T_c)})')
# plt.plot(bins_n[1:], cdf_n, label = f'NCC ({len(T_n)})')
# plt.xscale('log')
# plt.xlabel('T [keV]')
# plt.ylabel('CDF')
# plt.title('CDF for T')
# plt.legend(loc='best')
# #plt.savefig('CDF_R(cut)_ccVncc.png',dpi = 300)
# plt.show()
# general_functions.calculate_ks_stat(T_c, T_n)
# 
# =============================================================================
# =============================================================================
# 
# C_c = c_new['c']
# err_c_c = c_new['e_c']
# z_c = ycept_c + Slope_c* log_T_new_c
# z_n = ycept_n + Slope_n* log_T_new_n
# 
# C_n = n_new['c']
# err_c_n = n_new['e_c']
# plt.errorbar(C_c,z_c-log_r_c, yerr = sigma_r_c, xerr= err_c_c ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(z_c)})')
# plt.errorbar(C_n,z_n-log_r_n, yerr = sigma_r_n, xerr= err_c_n ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(z_n)})')
# plt.ylim(-1,1)
# plt.xlim(-0.1, 0.8)
# plt.xlabel('Concentration')
# plt.ylabel('$\Delta log_{10}R$')
# plt.title('$R-T$ residuals')
# plt.legend(loc = 'best')
# plt.axhline(0, color = 'black')
# plt.axvline(0.18, color= 'blue', ls= '--', label='Threshold')
# plt.show()
# 
# =============================================================================

# =============================================================================
# ###########################################################################
#                             # New scaling relation after adding C
# ##############################################################################
# c_new = CC_clusters
# Z_c = (c_new['z'])
# E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
# 
# R_c = c_new['R kpc']
# R_min_c = c_new['Rmin kpc']
# R_max_c = c_new['Rmax kpc']
# R_new_c = (R_c/250) * (E_c**(1.9))
# log_r_c = np.log10(R_new_c)
# sigma_r_c = 0.4343 * (R_max_c-R_min_c)/(2*R_c)
# c_new.iloc[0]
# T_c = c_new['T']
# T_new_c = T_c/4.5
# log_T_c = np.log10(T_c)
# log_T_new_c = np.log10(T_new_c)
# sigma_T_c = 0.4343 * (c_new['Tmax']-c_new['Tmin'])/(2*T_c)
# c_c = c_new['c']/np.median(c_new['c'])
# e_c_c = c_new['e_c']
# log_c_c = np.log10(c_c)
# sigma_c_c = 0.4343 * e_c_c/c_c
# cov = np.cov(sigma_r_c,sigma_c_c)
# yarray = log_r_c + 0.39*log_c_c
# xarray = log_T_new_c
# yerr = np.sqrt( (sigma_r_c)**2 + (-0.39*sigma_c_c)**2 + 2*-0.39*cov[0][1] )
# xerr = sigma_T_c 
# test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# #Bootstrap
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = c_new.sample(n = len(c_new), replace = True)
# # 
# #     Z_c = (random_clusters['z'])
# #     E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
# #     
# #     R_c = random_clusters['R kpc']
# #     R_min_c = random_clusters['Rmin kpc']
# #     R_max_c = random_clusters['Rmax kpc']
# #     R_new_c = (R_c/250) * (E_c**(1.9))
# #     log_r_c = np.log10(R_new_c)
# #     sigma_r_c = 0.4343 * (R_max_c-R_min_c)/(2*R_c)
# #     
# #     T_c = random_clusters['T']
# #     T_new_c = T_c/4.5
# #     log_T_c = np.log10(T_new_c)
# #     log_T_new_c = np.log10(T_c/4.5)
# #     sigma_T_c = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T_c)
# #     c_c = random_clusters['c']/np.median(random_clusters['c'])
# #     e_c_c = random_clusters['e_c']
# #     log_c_c = np.log10(c_c)
# #     sigma_c_c = 0.4343 * e_c_c/c_c
# #     cov = np.cov(sigma_r_c,sigma_c_c)
# #     yarray = log_r_c - (-0.39)*log_c_c
# #     xarray = log_T_new_c
# #     yerr = np.sqrt( (sigma_r_c)**2 + (-0.39*sigma_c_c)**2 + 2*-0.39*cov[0][1] )
# #     xerr = sigma_T_c 
# #     test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# # 
# #     best_A.append(test_Norm_c)
# #     best_B.append(test_Slope_c)
# #     best_scatter.append(test_Scatter_c)
# # 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-T_CC(with_C)_BCES.csv')
# # =============================================================================
# 
# ###################    For NCC clusters ##############################
# n_new = NCC_clusters
# 
# Z_n = (n_new['z'])
# E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
# R_n = n_new['R kpc']
# R_min_n = n_new['Rmin kpc']
# R_max_n = n_new['Rmax kpc']
# R_new_n = (R_n/250) * (E_n**(1.9))
# log_r_n = np.log10(R_new_n)
# sigma_r_n = 0.4343 * (R_max_n-R_min_n)/(2*R_n)
# 
# T_n = n_new['T']
# T_new_n = T_n/4.5
# log_T_n = np.log10(T_n)
# log_T_new_n = np.log10(T_new_n)
# sigma_T_n = 0.4343 * (n_new['Tmax']-n_new['Tmin'])/(2*T_n)
# 
# c_n = n_new['c']/np.median(n_new['c'])
# e_c_n = n_new['e_c']
# log_c_n = np.log10(c_n)
# sigma_c_n = 0.4343 * e_c_n/c_n
# cov = np.cov(sigma_r_n,sigma_c_n)
# yarray = log_r_n - (-0.390)*log_c_n
# xarray = log_T_new_n
# yerr = np.sqrt( (sigma_r_n)**2 + (-0.39*sigma_c_n)**2 + 2*-0.39*cov[0][1] )
# xerr = sigma_T_n 
# test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = n_new.sample(n = len(n_new), replace = True)
# #     Z_n = (random_clusters['z'])
# #     E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
# #     R_n = random_clusters['R kpc']
# #     R_min_n = random_clusters['Rmin kpc']
# #     R_max_n = random_clusters['Rmax kpc']
# #     R_new_n = (R_n/250) * (E_n**(1.9))
# #     log_r_n = np.log10(R_new_n)
# #     sigma_r_n = 0.4343 * (R_max_n-R_min_n)/(2*R_n)
# #     
# #     T_n = random_clusters['T']
# #     T_new_n = T_n/4.5
# #     log_T_n = np.log10(T_n)
# #     log_T_new_n = np.log10(T_new_n)
# #     sigma_T_n = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T_n)
# #     
# #     c_n = random_clusters['c']/np.median(random_clusters['c'])
# #     e_c_n = random_clusters['e_c']
# #     log_c_n = np.log10(c_n)
# #     sigma_c_n = 0.4343 * e_c_n/c_n
# #     cov = np.cov(sigma_r_n,sigma_c_n)
# #     yarray = log_r_n - (-0.390)*log_c_n
# #     xarray = log_T_new_n
# #     yerr = np.sqrt( (sigma_r_n)**2 + (-0.39*sigma_c_n)**2 + 2*-0.39*cov[0][1] )
# #     xerr = sigma_T_n 
# #     test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# # 
# #     
# #     best_A.append(test_Norm_n)
# #     best_B.append(test_Slope_n)
# #     best_scatter.append(test_Scatter_n)
# # 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-T_NCC(with_C)_BCES.csv')
# # =============================================================================
# 
# 
# data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_CC(with_C)_BCES.csv')
# norm_c = data_c['Normalization']
# slope_c = data_c['Slope']
# scatter_c = data_c['Scatter']
# 
# errnorm_c = general_functions.calculate_asymm_err(norm_c)
# errslope_c = general_functions.calculate_asymm_err(slope_c)
# errscatter_c = general_functions.calculate_asymm_err(scatter_c)
# 
# data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_NCC(with_C)_BCES.csv')
# norm_n = data_n['Normalization']
# slope_n = data_n['Slope']
# scatter_n = data_n['Scatter']
# 
# errnorm_n = general_functions.calculate_asymm_err(norm_n)
# errslope_n = general_functions.calculate_asymm_err(slope_n)
# errscatter_n = general_functions.calculate_asymm_err(scatter_n)
# 
# print(f'Normalization sigma after including C: {general_functions.calculate_sigma_dev(test_Norm_c,errnorm_c,test_Norm_n,errnorm_n)}')
# print(f'Slope sigma after including C: {general_functions.calculate_sigma_dev(test_Slope_c,errslope_c,test_Slope_n,errslope_n)}')
# print(f'Scatter sigma after including C: {general_functions.calculate_sigma_dev(test_Scatter_c,errscatter_c,test_Scatter_n,errscatter_n)}')
# 
# plt.errorbar(log_T_c,log_r_c,yerr=sigma_r_c,xerr=sigma_T_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
# plt.errorbar(log_T_n,log_r_n,yerr=sigma_r_n,xerr=sigma_T_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_r_n)})')
# 
# #z = best_a + best_b*log_Lx
# z_c = test_Ycept_c + log_T_new_c*test_Slope_c
# z_n = test_Ycept_n + log_T_new_n*test_Slope_n
# 
# #plt.axvline(np.log10(0.15),color='Black')
# 
# plt.plot(log_T_c,z_c,label='Best fit CC',color='black')
# plt.plot(log_T_n,z_n,label='Best fit NCC',color='blue')
# 
# #plt.xlim(1.5,25)
# #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# plt.xlabel('$log_{10}(T$ [keV])')
# plt.ylabel(' $log_{10}((R/c^{-0.39})*E(z)^{1.9}$ / 250 [kpc])')
# plt.title('$R-T.c$ best fit  ')
# plt.legend(loc='best')
# #plt.savefig('R-Lx_best_fit-CCvNCC().png',dpi=300)
# plt.show()
# 
# 
# =============================================================================

