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
bestfit_Norm = bestfit_values['Norm_all'][4]
err_bestfit_Norm = bestfit_values['err_Norm_all'][4]
bestfit_Slope = bestfit_values['Slope_all'][4]
err_bestfit_Slope = bestfit_values['err_Slope_all'][4]
bestfit_Scatter = bestfit_values['Scatter_all'][4]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][4]


bcgy = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL_mass.csv')
bcgy = general_functions.cleanup(bcgy)
bcgy = bcgy[(bcgy['z']>0.03) & (bcgy['z']< 0.15)] 
StN = bcgy['Y']/bcgy['eY']
bcgy = bcgy[ StN > 2]

L_sun = 1
Lbcg = L_sun * 10 ** (0.4*(3.27 - bcgy['BCGMag']))
bcgy['Lbcg'] = Lbcg

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)

bcgy_all = pd.merge(bcgy, thesis_table, left_on = bcgy['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')
g = bcgy_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# CC clusters
Z = (CC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin_c = CC_clusters['Y']
e_Y_arcmin_c = CC_clusters['eY']
Ysz_c = (Ysz_arcmin_c * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c/20
log_Ysz_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin_c/Ysz_arcmin_c
err_Ysz_c = ((e_Y_arcmin_c)* (D_a.value**2) * (np.pi / (60*180))**2)


Lbcg_c = CC_clusters['Lbcg']
log_Lbcg_c = np.log10(Lbcg_c)
Lbcg_new_c = Lbcg_c / 6e11
log_Lbcg_new_c = np.log10(Lbcg_new_c)
sigma_Lbcg_c = np.zeros(len(Lbcg_c))

ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_Lbcg_new_c,sigma_Lbcg_c,log_Ysz_c,sigma_Ysz_c)

# BCES bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# 
#     Lbcg =random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(Lbcg))
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_CC_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)

# NCC clusters

Z = (NCC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000
Ysz_arcmin_n = NCC_clusters['Y']
e_Y_arcmin_n = NCC_clusters['eY']
Ysz_n = (Ysz_arcmin_n * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n /20
log_Ysz_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin_n/Ysz_arcmin_n
err_Ysz_n = ((e_Y_arcmin_n)* (D_a.value**2) * (np.pi / (60*180))**2)

Lbcg_n = NCC_clusters['Lbcg']
log_Lbcg_n = np.log10(Lbcg_n)
Lbcg_new_n = Lbcg_n / 6e11
log_Lbcg_new_n = np.log10(Lbcg_new_n)
sigma_Lbcg_n = np.zeros(len(Lbcg_n))
ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Lbcg_new_n,sigma_Lbcg_n,log_Ysz_n,sigma_Ysz_n)

#Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# 
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
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_NCC_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_NCC_BCES.csv')
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



sns.set_context('paper')
Y_linspace = np.linspace(1,25,100)

z_c = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept_n, Slope_n)

plt.plot(Y_linspace ,z_c, label = 'Best fit CC',color = 'blue')
plt.plot(Y_linspace ,z_n, label = 'Best fit NCC',color = 'black')
plt.errorbar(Lbcg_c/1e11,Ysz_c, yerr = err_Ysz_c,color = 'green',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'CC Clusters ({len(Lbcg_c)})' )
plt.errorbar(Lbcg_n/1e11,Ysz_n, yerr = err_Ysz_n,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'NCC Clusters ({len(Lbcg_n)})' )
plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'lower right')
plt.xlabel('$L_{\mathrm{BCG}}$ (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.ylabel(' $Y_{\mathrm{SZ}}$  ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{\mathrm{SZ}}-L_{\mathrm{BCG}}$ best fit')
plt.xlim(1.3,25)
plt.ylim(0.5,1000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Ysz-Lbcg_ccVncc_bestfit.png',dpi=300, bbox_inches="tight")
plt.show()

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')

print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))






## Cutting galaxy groups based on Mass  #############

bestfit_Norm_clusters = bestfit_values['Norm_clusters'][4]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][4]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][4]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][4]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][4]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][4]

# CC clusters
c_new = general_functions.removing_galaxy_groups(CC_clusters)
Z = (c_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000

Ysz_arcmin_c = c_new['Y']
e_Y_arcmin_c = c_new['eY']
Ysz_c = (Ysz_arcmin_c * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c/20
log_Ysz_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin_c/Ysz_arcmin_c
err_Ysz_c = ((e_Y_arcmin_c)* (D_a.value**2) * (np.pi / (60*180))**2)


Lbcg_c = c_new['Lbcg']
log_Lbcg_c = np.log10(Lbcg_c)
Lbcg_new_c = Lbcg_c / 6e11
log_Lbcg_new_c = np.log10(Lbcg_new_c)
sigma_Lbcg_c = np.zeros(len(Lbcg_c))

ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_c,sigma_Lbcg_c,log_Ysz_c,sigma_Ysz_c)

# BCES bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
#     
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# 
#     Lbcg =random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(Lbcg))
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Ysz,sigma_Ysz)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_CC(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_CC(Mcut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)

# NCC clusters
n_new = general_functions.removing_galaxy_groups(NCC_clusters)
Z = (n_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000
Ysz_arcmin_n = n_new['Y']
e_Y_arcmin_n = n_new['eY']
Ysz_n = (Ysz_arcmin_n * (D_a.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n /20
log_Ysz_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin_n/Ysz_arcmin_n
err_Ysz_n = ((e_Y_arcmin_n)* (D_a.value**2) * (np.pi / (60*180))**2)

Lbcg_n = n_new['Lbcg']
log_Lbcg_n = np.log10(Lbcg_n)
Lbcg_new_n = Lbcg_n / 6e11
log_Lbcg_new_n = np.log10(Lbcg_new_n)
sigma_Lbcg_n = np.zeros(len(Lbcg_n))
ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_n,sigma_Lbcg_n,log_Ysz_n,sigma_Ysz_n)

#Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
#     Ysz_arcmin =random_clusters['Y']
#     e_Y_arcmin = random_clusters['eY']
#     Ysz = (Ysz_arcmin * (D_a**2) * (np.pi / (60*180))**2)
#     Ysz_new = Ysz /20
#     log_Ysz = np.log10(Ysz_new)
#     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# 
# 
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
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Ysz-Lbcg_NCC(Mcut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Ysz-Lbcg_NCC(Mcut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut =  general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)

sns.set_context('paper')
Y_linspace = np.linspace(1,25,100)

z_c = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(Y_linspace, 6, 20, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(Y_linspace ,z_c, label = 'Best fit CC',color = 'blue')
plt.plot(Y_linspace ,z_n, label = 'Best fit NCC',color = 'black')
plt.errorbar(Lbcg_c/1e11,Ysz_c, yerr = err_Ysz_c,color = 'green',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'CC Clusters ({len(Lbcg_c)})' )
plt.errorbar(Lbcg_n/1e11,Ysz_n, yerr = err_Ysz_n,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'NCC Clusters ({len(Lbcg_n)})' )
plt.xscale('log')
plt.yscale('log')

plt.legend(loc = 'lower right')
plt.xlabel('$L_{\mathrm{BCG}}$ (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.ylabel(' $Y_{\mathrm{SZ}}$  ($\mathrm{kpc}^{2}$)')
plt.title('$Y_{\mathrm{SZ}}-L_{\mathrm{BCG}}$ best fit')
plt.xlim(1.3,25)
plt.ylim(0.5,1000)
plt.title('$Y_{\mathrm{SZ}}-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Ysz-Lbcg_ccVncc_Mcut_bestfit.png',dpi=300, bbox_inches="tight")
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
    
plt.xlim(-0.65,3)
plt.ylim(0.2,2.5)
plt.legend(prop = {'size' : 8},loc='lower right')
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$Y_{SZ} - L_{BCG}$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Ysz-Lbcg_ccVncc_contours.png', dpi = 300, bbox_inches="tight")

plt.show()



