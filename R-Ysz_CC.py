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
bestfit_Norm = bestfit_values['Norm_all'][7]
err_bestfit_Norm = bestfit_values['err_Norm_all'][7]
bestfit_Slope = bestfit_values['Slope_all'][7]
err_bestfit_Slope = bestfit_values['err_Slope_all'][7]
bestfit_Scatter = bestfit_values['Scatter_all'][7]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][7]


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

#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
ry['R kpc'] = R_kpc
ry['Rmin kpc'] = Rmin_kpc
ry['Rmax kpc'] = Rmax_kpc



thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
ry_all = pd.merge(ry, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')
g = ry_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')


# CC clusters
Z = (CC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R_c = CC_clusters['R kpc']
R_min_c = CC_clusters['Rmin kpc']
R_max_c = CC_clusters['Rmax kpc'] 
r_new_c = (R_c/250) * (E**(0.87))
log_r_c = np.log10(r_new_c)
sigma_r_c = 0.4343 * (R_max_c - R_min_c)/(2*R_c)
err_r_c = [R_c-R_min_c,R_max_c-R_c]


Ysz_arcmin_c = CC_clusters['Ysz']
e_Y_arcmin_c = CC_clusters['eY']
Ysz_c = (Ysz_arcmin_c * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c /25
log_Ysz_c = np.log10(Ysz_c)
log_Ysz_new_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin_c/Ysz_arcmin_c
err_Ysz_c = ((e_Y_arcmin_c)* (d_A.value**2) * (np.pi / (60*180))**2)
ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_Ysz_new_c,sigma_Ysz_c,log_r_c,sigma_r_c)

# Bootstrap : BCES

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
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
# bestfit_bootstrap.to_csv('R-Ysz_CC_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)


# FOR NCC clusters
Z = (NCC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R_n = NCC_clusters['R kpc']
R_min_n = NCC_clusters['Rmin kpc']
R_max_n = NCC_clusters['Rmax kpc'] 
r_new_n = (R_n/250) * (E**(0.87))
log_r_n = np.log10(r_new_n)
sigma_r_n = 0.4343 * (R_max_n - R_min_n)/(2*R_n)
err_r_n = [R_n-R_min_n,R_max_n-R_n]



Ysz_arcmin_n = NCC_clusters['Ysz']
e_Y_arcmin_n = NCC_clusters['eY']
Ysz_n = (Ysz_arcmin_n * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n /25
log_Ysz_n = np.log10(Ysz_n)
log_Ysz_new_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin_n/Ysz_arcmin_n
err_Ysz_n = ((e_Y_arcmin_n)* (d_A.value**2) * (np.pi / (60*180))**2)
ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Ysz_new_n,sigma_Ysz_n,log_r_n,sigma_r_n)


##Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# 
# 
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     d_A = cosmo.angular_diameter_distance(Z)*1000
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
# bestfit_bootstrap.to_csv('R-Ysz_NCC_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_NCC_BCES.csv')
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
Ysz_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept_n, Slope_n)

plt.plot(Ysz_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(Ysz_linspace,z_n,label='Best fit NCC', color ='black')

plt.errorbar(Ysz_c,R_c,yerr=err_r_c,xerr=err_Ysz_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
plt.errorbar(Ysz_n,R_n,yerr=err_r_n,xerr = err_Ysz_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(Ysz_n)})')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.3,1500)
plt.ylim(15,3000)
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(r'$Y_{\mathrm{SZ}} \,(\mathrm{kpc}^{2}$) ')
plt.ylabel(r' $R*E(z)^{0.87}$ (kpc)')
plt.title(r'$R-Y_{\mathrm{SZ}}$ best fit ')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Ysz_ccVncc_bestfit.png',dpi=300,bbox_inches='tight')
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



bestfit_Norm_clusters = bestfit_values['Norm_clusters'][7]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][7]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][7]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][7]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][7]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][7]




# CC clusters
c_new = general_functions.removing_galaxy_groups(CC_clusters)
Z = (c_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R_c = c_new['R kpc']
R_min_c = c_new['Rmin kpc']
R_max_c = c_new['Rmax kpc'] 
r_new_c = (R_c/250) * (E**(0.87))
log_r_c = np.log10(r_new_c)
sigma_r_c = 0.4343 * (R_max_c - R_min_c)/(2*R_c)
err_r_c = [R_c-R_min_c,R_max_c-R_c]


Ysz_arcmin_c = c_new['Ysz']
e_Y_arcmin_c = c_new['eY']
Ysz_c = (Ysz_arcmin_c * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new_c = Ysz_c /25
log_Ysz_c = np.log10(Ysz_c)
log_Ysz_new_c = np.log10(Ysz_new_c)
sigma_Ysz_c  =  0.4343*e_Y_arcmin_c/Ysz_arcmin_c
err_Ysz_c = ((e_Y_arcmin_c)* (d_A.value**2) * (np.pi / (60*180))**2)
ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_Ysz_new_c,sigma_Ysz_c,log_r_c,sigma_r_c)

# Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
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
# bestfit_bootstrap.to_csv('R-Ysz_CC(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_CC(Mcut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)


# FOR NCC clusters
n_new = general_functions.removing_galaxy_groups(NCC_clusters)
Z = (n_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000


R_n = n_new['R kpc']
R_min_n = n_new['Rmin kpc']
R_max_n = n_new['Rmax kpc'] 
r_new_n = (R_n/250) * (E**(0.87))
log_r_n = np.log10(r_new_n)
sigma_r_n = 0.4343 * (R_max_n - R_min_n)/(2*R_n)
err_r_n = [R_n-R_min_n, R_max_n-R_n]



Ysz_arcmin_n = n_new['Ysz']
e_Y_arcmin_n = n_new['eY']
Ysz_n = (Ysz_arcmin_n * (d_A.value**2) * (np.pi / (60*180))**2)
Ysz_new_n = Ysz_n /25
log_Ysz_n = np.log10(Ysz_n)
log_Ysz_new_n = np.log10(Ysz_new_n)
sigma_Ysz_n  =  0.4343*e_Y_arcmin_n/Ysz_arcmin_n
err_Ysz_n = ((e_Y_arcmin_n)* (d_A.value**2) * (np.pi / (60*180))**2)
ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_Ysz_new_n,sigma_Ysz_n,log_r_n,sigma_r_n)


##Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
#     omega_m = 0.3
#     omega_lambda = 0.7
#     cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     d_A = cosmo.angular_diameter_distance(Z)*1000
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
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Y_NCC(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Ysz_NCC(Mcut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut =  general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)


sns.set_context('paper')
Ysz_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(Ysz_linspace, 25, 250, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(Ysz_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(Ysz_linspace,z_n,label='Best fit NCC', color ='black')

plt.errorbar(Ysz_c,R_c,yerr=err_r_c,xerr=err_Ysz_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
plt.errorbar(Ysz_n,R_n,yerr=err_r_n,xerr = err_Ysz_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(Ysz_n)})')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.3,1500)
plt.ylim(15,3000)
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(r'$Y_{\mathrm{SZ}} \,(\mathrm{kpc}^{2}$) ')
plt.ylabel(r' $R*E(z)^{0.87}$ (kpc)')
plt.title(r'$R-Y_{\mathrm{SZ}}$ best fit $(M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Ysz_ccVncc_bestfit_Mcut.png',dpi=300,bbox_inches='tight')
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
    
plt.xlim(-0.2,0.5)
plt.ylim(0.5,1.6)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-Y_{\mathrm{SZ}}$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Ysz_ccVncc_contours.png' ,dpi=300, bbox_inches="tight")

plt.show()


# =============================================================================
# Y_c = Ysz_c
# Y_n = Ysz_n
# bins_c, cdf_c = general_functions.calculate_cdf(Y_c, 20)
# bins_n, cdf_n = general_functions.calculate_cdf(Y_n, 20)
# plt.plot(bins_c[1:], cdf_c,label = f'CC ({len(Y_c)})')
# plt.plot(bins_n[1:], cdf_n, label = f'NCC ({len(Y_n)})')
# plt.xscale('log')
# plt.xlabel('Y [$kpc^{2}$]')
# plt.ylabel('CDF')
# plt.title('CDF for Ysz')
# plt.legend(loc='best')
# #plt.savefig('CDF_R(cut)_ccVncc.png',dpi = 300)
# plt.show()
# general_functions.calculate_ks_stat(Y_c, Y_n)
# 
# =============================================================================
##Residuals against Concentration

# =============================================================================
# C_c = CC_clusters['c']
# err_c_c = CC_clusters['e_c']
# z_c = ycept_c + Slope_c* log_Ysz_new_c
# z_n = ycept_n + Slope_n* log_Ysz_new_n
# 
# C_n = NCC_clusters['c']
# err_c_n = NCC_clusters['e_c']
# plt.errorbar(C_c,z_c-log_r_c, yerr = sigma_r_c, xerr= err_c_c ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(z_c)})')
# plt.errorbar(C_n,z_n-log_r_n, yerr = sigma_r_n, xerr= err_c_n ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(z_n)})')
# plt.ylim(-1,1)
# plt.xlim(-0.1, 0.8)
# plt.xlabel('Concentration')
# plt.ylabel('$\Delta log_{10}R$')
# plt.title('$R-Y_{SZ}$ residuals')
# plt.legend(loc = 'best')
# plt.axhline(0, color = 'black')
# plt.axvline(0.18, color= 'blue', ls= '--', label='Threshold')
# plt.show()
# =============================================================================

###########################################################################
                            # New scaling relation after adding C
##############################################################################


# =============================================================================
# Z = (CC_clusters['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# d_A = cosmo.angular_diameter_distance(Z)*1000
# R_c = CC_clusters['R kpc']
# R_min_c = CC_clusters['Rmin kpc']
# R_max_c = CC_clusters['Rmax kpc'] 
# r_new_c = (R_c/250) * (E**(-0.17))
# log_r_c = np.log10(r_new_c)
# sigma_r_c = 0.4343 * (R_max_c - R_min_c)/(2*R_c)
# err_r_c = [(R_c-R_min_c)/250,(R_max_c-R_c)/250]
# 
# Ysz_arcmin_c = CC_clusters['Ysz']
# e_Y_arcmin_c = CC_clusters['eY']
# Ysz_c = (Ysz_arcmin_c * (d_A.value**2) * (np.pi / (60*180))**2)
# Ysz_new_c = Ysz_c /25
# log_Ysz_c = np.log10(Ysz_c)
# log_Ysz_new_c = np.log10(Ysz_new_c)
# sigma_Ysz_c  =  0.4343*e_Y_arcmin_c/Ysz_arcmin_c
# err_Ysz_c = ((e_Y_arcmin_c)* (d_A.value**2) * (np.pi / (60*180))**2)
# 
# c_c = CC_clusters['c']/np.median(CC_clusters['c'])
# e_c_c = CC_clusters['e_c']
# log_c_c = np.log10(c_c)
# sigma_c_c = 0.4343 * e_c_c/c_c
# cov = np.cov(sigma_r_c,sigma_c_c)
# yarray = log_r_c - (-0.380)*log_c_c
# xarray = log_Ysz_new_c
# yerr = np.sqrt( (sigma_r_c)**2 + (-0.38*sigma_c_c)**2 + 2*-0.38*cov[0][1] )
# xerr = sigma_Ysz_c 
# test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# #Bootstrap
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     d_A = cosmo.angular_diameter_distance(Z)*1000
# # 
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc'] 
# #     r_new = (R/250) * (E**(-0.17))
# #     log_r = np.log10(r_new)
# #     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# # 
# #     Ysz_arcmin = random_clusters['Ysz']
# #     e_Y_arcmin = random_clusters['eY']
# #     Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
# #     Ysz_new = Ysz /25
# #     log_Ysz = np.log10(Ysz)
# #     log_Ysz_new = np.log10(Ysz_new)
# #     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# #     c = random_clusters['c']/np.median(random_clusters['c'])
# #     e_c = random_clusters['e_c']
# #     log_c = np.log10(c)
# #     sigma_c = 0.4343 * e_c/c
# #     cov = np.cov(sigma_r,sigma_c)
# #     yarray = log_r - (-0.380)*log_c
# #     xarray = log_Ysz_new
# #     yerr = np.sqrt( (sigma_r)**2 + (-0.38*sigma_c)**2 + 2*-0.38*cov[0][1] )
# #     xerr = sigma_Ysz 
# #     test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# # 
# # 
# #     best_A.append(test_Norm)
# #     best_B.append(test_Slope)
# #     best_scatter.append(test_Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Y.c_CC_BCES.csv')
# # 
# # =============================================================================
# Z = (NCC_clusters['z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# d_A = cosmo.angular_diameter_distance(Z)*1000
# R_n = NCC_clusters['R kpc']
# R_min_n = NCC_clusters['Rmin kpc']
# R_max_n = NCC_clusters['Rmax kpc'] 
# r_new_n = (R_n/250) * (E**(0.48))
# log_r_n = np.log10(r_new_n)
# sigma_r_n = 0.4343 * (R_max_n - R_min_n)/(2*R_n)
# 
# Ysz_arcmin_n = NCC_clusters['Ysz']
# e_Y_arcmin_n = NCC_clusters['eY']
# Ysz_n = (Ysz_arcmin_n * (d_A.value**2) * (np.pi / (60*180))**2)
# Ysz_new_n = Ysz_n /45
# log_Ysz_n = np.log10(Ysz_n)
# log_Ysz_new_n = np.log10(Ysz_new_n)
# sigma_Ysz_n  =  0.4343*e_Y_arcmin_n/Ysz_arcmin_n
# c_n = NCC_clusters['c']/np.median(NCC_clusters['c'])
# e_c_n = NCC_clusters['e_c']
# log_c_n = np.log10(c_n)
# sigma_c_n = 0.4343 * e_c_n/c_n
# cov = np.cov(sigma_r_n,sigma_c_n)
# yarray = log_r_n - (-0.380)*log_c_n
# xarray = log_Ysz_new_n
# yerr = np.sqrt( (sigma_r_n)**2 + (-0.38*sigma_c_n)**2 + 2*-0.38*cov[0][1] )
# xerr = sigma_Ysz_n 
# test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     d_A = cosmo.angular_diameter_distance(Z)*1000
# # 
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc'] 
# #     r_new = (R/250) * (E**(-1.08))
# #     log_r = np.log10(r_new)
# #     sigma_r = 0.4343 * (R_max - R_min)/(2*R)
# # 
# #     Ysz_arcmin = random_clusters['Ysz']
# #     e_Y_arcmin = random_clusters['eY']
# #     Ysz = (Ysz_arcmin * (d_A.value**2) * (np.pi / (60*180))**2)
# #     Ysz_new = Ysz /45
# #     log_Ysz = np.log10(Ysz)
# #     log_Ysz_new = np.log10(Ysz_new)
# #     sigma_Ysz  =  0.4343*e_Y_arcmin/Ysz_arcmin
# #     c = random_clusters['c']/np.median(random_clusters['c'])
# #     e_c = random_clusters['e_c']
# #     log_c = np.log10(c)
# #     sigma_c = 0.4343 * e_c/c
# #     cov = np.cov(sigma_r,sigma_c)
# #     yarray = log_r - (-0.380)*log_c
# #     xarray = log_Ysz_new
# #     yerr = np.sqrt( (sigma_r)**2 + (-0.38*sigma_c)**2 + 2*-0.38*cov[0][1] )
# #     xerr = sigma_Ysz 
# #     test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# # 
# # 
# #     best_A.append(test_Norm)
# #     best_B.append(test_Slope)
# #     best_scatter.append(test_Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Y.c_NCC_BCES.csv')
# # 
# # =============================================================================
# data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/py_scripts/R-Y.c_CC_BCES.csv')
# norm_c = data_c['Normalization']
# slope_c = data_c['Slope']
# scatter_c = data_c['Scatter']
# 
# errnorm_c = general_functions.calculate_asymm_err(norm_c)
# errslope_c = general_functions.calculate_asymm_err(slope_c)
# errscatter_c = general_functions.calculate_asymm_err(scatter_c)
# 
# data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/py_scripts/R-Y.c_NCC_BCES.csv')
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
# =============================================================================




