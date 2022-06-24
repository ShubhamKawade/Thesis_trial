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
bestfit_Norm = bestfit_values['Norm_all'][8]
err_bestfit_Norm = bestfit_values['err_Norm_all'][8]
bestfit_Slope = bestfit_values['Slope_all'][8]
err_bestfit_Slope = bestfit_values['err_Slope_all'][8]
bestfit_Scatter = bestfit_values['Scatter_all'][8]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][8]



r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')
r.rename({'# Name':'Cluster'},axis=1,inplace=True)
r = general_functions.cleanup(r)

bcg = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIFL-BCG-2MASS-FINAL.csv')
bcg.rename({'#Cluster':'Cluster'},axis=1,inplace=True)
bcg = general_functions.cleanup(bcg)
bcg = bcg[bcg['z']<0.15]
bcg = bcg[bcg['z']>0.03] 

r = r[r['R'] > 2]
R_old = r['R']


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
#for i in range(len(theta)):
 ##   R_k = (theta[i] * d_A[i])
   # R_kpc.append(R_k.value)

#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
r['R kpc'] = R_kpc
r['Rmin kpc'] = Rmin_kpc
r['Rmax kpc'] = Rmax_kpc

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)

#rbcg= pd.merge(bcg, r, left_on = bcg['Cluster'].str.casefold(), right_on = r['Cluster'].str.casefold(), how ='inner')
#rbcg.to_csv('/home/schubham/Thesis/Thesis/Data/rbcg_temp.csv')
rbcg = pd.read_csv('/home/schubham/Thesis/Thesis/Data/rbcg_temp.csv')
rlbcg_all = pd.merge(rbcg, thesis_table, left_on = rbcg['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')
g = rlbcg_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

#CC clusters
Z = (CC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R_c = CC_clusters['R kpc']
R_min_c = CC_clusters['Rmin kpc']
R_max_c = CC_clusters['Rmax kpc']
R_new_c = (R_c/250) *E ** (-2.04)
log_r_c = np.log10(R_new_c)
err_r_c = [R_new_c-R_min_c,R_max_c-R_new_c]
sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))


Lbcg_c = CC_clusters['L_bcg(1e11solar)']
log_Lbcg_c = np.log10(Lbcg_c)
Lbcg_new_c = Lbcg_c / 6
log_Lbcg_new_c = np.log10(Lbcg_new_c)
sigma_Lbcg_c = np.zeros(len(Lbcg_c))
ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_Lbcg_new_c,sigma_Lbcg_c,log_r_c,sigma_r_c)

# Bootstrap for CC
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
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
#      Lbcg = random_clusters['L_bcg (1e11 solar)']
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
# bestfit_bootstrap.to_csv('R-Lbcg_CC_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)


# # NCC CLUSTERS

Z = (NCC_clusters['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R_n = NCC_clusters['R kpc']
R_min_n = NCC_clusters['Rmin kpc']
R_max_n = NCC_clusters['Rmax kpc']
R_new_n = (R_n/250) * E ** -2.04
log_r_n = np.log10(R_new_n)
err_r_n = [R_new_n-R_min_n,R_max_n-R_new_n]
sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))


Lbcg_n = NCC_clusters['L_bcg(1e11solar)']
log_Lbcg_n = np.log10(Lbcg_n)
Lbcg_new_n = Lbcg_n / 6
log_Lbcg_new_n = np.log10(Lbcg_new_n)
sigma_Lbcg_n = np.zeros(len(Lbcg_n))

ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Lbcg_new_n,sigma_Lbcg_n,log_r_n,sigma_r_n)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
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
#      R_new = (R/250)*E**-2.04
#      log_r = np.log10(R_new)
#      sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#      Lbcg = random_clusters['L_bcg (1e11 solar)']
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
# bestfit_bootstrap.to_csv('R-Lbcg_NCC_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_NCC_BCES.csv')
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


err_r_c = [R_c-R_min_c,R_max_c-R_c]
err_r_n = [R_n-R_min_n,R_max_n-R_n]

sns.set_context('paper')

plt.errorbar(Lbcg_c,R_c,yerr=err_r_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_Lbcg_c)})')
plt.errorbar(Lbcg_n,R_n,yerr=err_r_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_Lbcg_n)})')

sns.set_context('paper')
Lbcg_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept_n, Slope_n)

plt.plot(Lbcg_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(Lbcg_linspace,z_n,label='Best fit NCC', color ='black')


plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{BCG}}$ ($10^{11} \,\mathrm{L}_{\odot}$)')
plt.ylabel(' $R*E(z)^{-2.04}$ (kpc)')
plt.title('$R-L_{\mathrm{BCG}}$ best fit')
plt.legend( loc='lower right')
plt.xlim(0.8,35)
plt.ylim(30,2000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Lbcg_ccVncc_bestfit.png',dpi=300,bbox_inches='tight')
plt.show()


# Plotting the residuals
print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')


print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))




####################################################################3
         # Removing galaxy groups based on mass cut
####################################################################3



bestfit_Norm_clusters = bestfit_values['Norm_clusters'][8]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][8]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][8]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][8]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][8]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][8]



#CC clusters
c_new = general_functions.removing_galaxy_groups(CC_clusters)
Z = (c_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R_c = c_new['R kpc']
R_min_c = c_new['Rmin kpc']
R_max_c = c_new['Rmax kpc']
R_new_c = (R_c/250) *E ** (-2.04)
log_r_c = np.log10(R_new_c)
err_r_c = [R_new_c-R_min_c,R_max_c-R_new_c]
sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))


Lbcg_c = c_new['L_bcg(1e11solar)']
log_Lbcg_c = np.log10(Lbcg_c)
Lbcg_new_c = Lbcg_c / 6
log_Lbcg_new_c = np.log10(Lbcg_new_c)
sigma_Lbcg_c = np.zeros(len(Lbcg_c))
ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_c,sigma_Lbcg_c,log_r_c,sigma_r_c)

# Bootstrap for CC
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = c_new.sample(n = len(c_new), replace = True)
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
#      Lbcg = random_clusters['L_bcg (1e11 solar)']
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
# bestfit_bootstrap.to_csv('R-Lbcg_CC(Mcut)_BCES.csv')
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_CC(Mcut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)


# # NCC CLUSTERS
n_new = general_functions.removing_galaxy_groups(NCC_clusters)
Z = (n_new['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R_n = n_new['R kpc']
R_min_n = n_new['Rmin kpc']
R_max_n = n_new['Rmax kpc']
R_new_n = (R_n/250) * E ** -2.04
log_r_n = np.log10(R_new_n)
err_r_n = [R_new_n-R_min_n,R_max_n-R_new_n]
sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))

Lbcg_n = n_new['L_bcg(1e11solar)']
log_Lbcg_n = np.log10(Lbcg_n)
Lbcg_new_n = Lbcg_n / 6
log_Lbcg_new_n = np.log10(Lbcg_new_n)
sigma_Lbcg_n = np.zeros(len(Lbcg_n))

ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_n,sigma_Lbcg_n,log_r_n,sigma_r_n)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#      random_clusters = n_new.sample(n = len(n_new), replace = True)
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
#      R_new = (R/250)*E**-2.04
#      log_r = np.log10(R_new)
#      sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# 
# 
#      Lbcg = random_clusters['L_bcg (1e11 solar)']
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
# bestfit_bootstrap.to_csv('R-Lbcg_NCC(Mcut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lbcg_NCC(Mcut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut =  general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)


err_r_c = [R_c-R_min_c,R_max_c-R_c]
err_r_n = [R_n-R_min_n,R_max_n-R_n]

sns.set_context('paper')
plt.errorbar(Lbcg_c,R_c,yerr=err_r_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_Lbcg_c)})')
plt.errorbar(Lbcg_n,R_n,yerr=err_r_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_Lbcg_n)})')

Lbcg_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(Lbcg_linspace, 6, 250, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(Lbcg_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(Lbcg_linspace,z_n,label='Best fit NCC', color ='black')


plt.xscale('log')
plt.yscale('log')
plt.xlim(0.8,35)
plt.ylim(30,2000)#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{BCG}}$ ($10^{11} \,\mathrm{L}_{\odot}$)')
plt.ylabel(' $R*E(z)^{-2.04}$ (kpc)')
plt.title('$R-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc = 'lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Lbcg_ccVncc_bestfit_Mcut.png',dpi=300,bbox_inches='tight')
plt.show()


# Plotting the residuals
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
    
plt.xlim(-0.4,0.8)
plt.ylim(0.4,1.8)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-L_{\mathrm{BCG}}$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Lbcg_ccVncc_contours.png' ,dpi=300, bbox_inches="tight")

plt.show()




