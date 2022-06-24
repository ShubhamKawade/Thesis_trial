#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns
from matplotlib import rc  

#Importing the best fit values of the scaling relation fit
bestfit_values = pd.read_csv('/home/schubham/Thesis/Thesis/Data/best_fit_parameters.csv')
bestfit_Norm = bestfit_values['Norm_all'][6]
err_bestfit_Norm = bestfit_values['err_Norm_all'][6]
bestfit_Slope = bestfit_values['Slope_all'][6]
err_bestfit_Slope = bestfit_values['err_Slope_all'][6]
bestfit_Scatter = bestfit_values['Scatter_all'][6]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][6]

## Only eeHIF values are taken
r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')
r.rename({'# Name':'Cluster'},axis=1,inplace=True)
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


#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
rl['R kpc'] = R_kpc
rl['Rmin kpc'] = Rmin_kpc
rl['Rmax kpc'] = Rmax_kpc


thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rl_all = pd.merge(rl, thesis_table, left_on = rl['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')
g = rl_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

Z_c = (CC_clusters['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
R_c = CC_clusters['R kpc']
R_min_c = CC_clusters['Rmin kpc']
R_max_c = CC_clusters['Rmax kpc']
R_new_c = (R_c/250) * (E_c**(-0.68))
log_r_c = np.log10(R_new_c)
sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))
err_r_c = [R_c - R_min_c, R_max_c-R_c]

Lx_c = CC_clusters['Lx']
log_Lx_c = np.log10(Lx_c)
sigma_Lx_c = 0.4343*CC_clusters['eL']/100
err_Lx_c = CC_clusters['eL']*Lx_c/100
ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_Lx_c,sigma_Lx_c,log_r_c,sigma_r_c)


# =============================================================================
# z_c = ycept_c + Slope_c * log_Lx_c
# plt.errorbar(R_new_c,z_c-log_r_c, yerr=sigma_r_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red')
# plt.axhline(0)
# #plt.xlim(-1.5,1.2)
# plt.ylim(2.5,-2.5)
# plt.show()
# =============================================================================


# ### Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
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
#     
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# 
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_CC_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_CC_BCES.csv')
norm_c = data['Normalization']
slope_c = data['Slope']
scatter_c = data['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c =  general_functions.calculate_asymm_err(scatter_c)

# ### NCC Clusters
Z_n = (NCC_clusters['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
R_n = NCC_clusters['R kpc']
R_min_n = NCC_clusters['Rmin kpc']
R_max_n = NCC_clusters['Rmax kpc']
R_new_n = (R_n/250) * (E_n**(-0.68))
log_r_n = np.log10(R_new_n)
sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))
err_r_n = [R_n - R_min_n, R_max_n-R_n]

Lx_n = NCC_clusters['Lx']
log_Lx_n = np.log10(Lx_n)
sigma_Lx_n = 0.4343*NCC_clusters['eL']/100
err_Lx_n = NCC_clusters['eL']*Lx_n/100
ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Lx_n,sigma_Lx_n,log_r_n,sigma_r_n)


# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
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
#     
# 
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# 
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_NCC_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_NCC_BCES.csv')
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



plt.errorbar(Lx_c,R_c,yerr=err_r_c,xerr=err_Lx_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
plt.errorbar(Lx_n,R_n,yerr=err_r_n,xerr = err_Lx_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(Lx_n)})')

#z = best_a + best_b*log_Lx
sns.set_context('paper')
Lx_linspace = np.linspace(0.0001,125,100)
z_c = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_n, Slope_n)

plt.plot(Lx_linspace,z_c,label='Best fit CC',color='black')
plt.plot(Lx_linspace,z_n,label='Best fit NCC',color='blue')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.title(r'$R-L_{\mathrm{X}}$ best fit')
plt.legend(loc = 'lower right')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Lx_ccVncc_bestfit.png',dpi=300,bbox_inches='tight')
plt.show()


print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')

print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))



sns.set_context('paper')
fig, ax_plot = plt.subplots()
#ax.scatter(slope_c,norm_c)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=1,label=r'Relaxed contours', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_c,Norm_c,color = 'green', label='relaxed bestfit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=1,label=r'Disturbed contours', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_n,Norm_n,color = 'darkorange', label = 'disturbed best fit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')


plt.xlim(-0.05,0.3)
plt.ylim(0.2,1.8)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-L_{X}$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC clusters')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Lx_ccVncc_contours_full_sample.png' ,dpi=300, bbox_inches="tight")
plt.show()



####################################################################3
         # Removing galaxy groups based on mass cut
####################################################################3


bestfit_Norm_clusters = bestfit_values['Norm_clusters'][6]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][6]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][6]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][6]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][6]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][6]


c_new = general_functions.removing_galaxy_groups(CC_clusters)
Z_c = (c_new['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
R_c = c_new['R kpc']
R_min_c = c_new['Rmin kpc']
R_max_c = c_new['Rmax kpc']
R_new_c = (R_c/250) * (E_c**(-0.68))
log_r_c = np.log10(R_new_c)
sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))
err_r_c = [(R_c - R_min_c)/250, (R_max_c-R_c)/250]

Lx_c = c_new['Lx']
log_Lx_c = np.log10(Lx_c)
sigma_Lx_c = 0.4343*c_new['eL']/100
err_Lx_c = c_new['eL']*Lx_c/100
ycept_c_Mcut,Norm_c_Mcut,Slope_c_Mcut,Scatter_c_Mcut = general_functions.calculate_bestfit(log_Lx_c,sigma_Lx_c,log_r_c,sigma_r_c)


# =============================================================================
# plt.errorbar(Lx_c,R_new_c,yerr=err_r_c,xerr=err_Lx_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_Lx_c)})')
# z_c =Norm_c * Lx_c**Slope_c
# plt.plot(Lx_c,z_c,label='Best fit (Chisq = 0.68)')
# #plt.ylim(-1.8,1.7)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$log_{10}$( $L_{X} / 10^{44} erg/s)$')
# plt.ylabel('$log_{10}$( R /250 kpc)')
# plt.title('$R-L_{X}$ best fit for CC clusters ')
# plt.legend(bbox_to_anchor=[0.7,0.2])
# #plt.savefig('R-Lx_best_fit-CC.png',dpi=300)
# plt.show()
# =============================================================================


# ### Bootstrap : BCES
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
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
# bestfit_bootstrap.to_csv('R-Lx_CC(Mcut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_CC(Mcut)_BCES.csv')
norm_c_Mcut = data['Normalization']
slope_c_Mcut = data['Slope']
scatter_c_Mcut = data['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut =  general_functions.calculate_asymm_err(scatter_c_Mcut)

# ### NCC Clusters
n_new = general_functions.removing_galaxy_groups(NCC_clusters)
Z_n = (n_new['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
R_n = n_new['R kpc']
R_min_n = n_new['Rmin kpc']
R_max_n = n_new['Rmax kpc']
R_new_n = (R_n/250) * (E_n**(-0.68))
log_r_n = np.log10(R_new_n)
sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))
err_r_n = [R_n - R_min_n, R_max_n-R_n]

Lx_n = n_new['Lx']
log_Lx_n = np.log10(Lx_n)
sigma_Lx_n = 0.4343*n_new['eL']/100
err_Lx_n = n_new['eL']*Lx_n/100
ycept_n_Mcut,Norm_n_Mcut,Slope_n_Mcut,Scatter_n_Mcut = general_functions.calculate_bestfit(log_Lx_n,sigma_Lx_n,log_r_n,sigma_r_n)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
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
# bestfit_bootstrap.to_csv('R-Lx_NCC(Mcut)_BCES.csv')
# 
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_NCC(Mcut)_BCES.csv')
norm_n_Mcut = data['Normalization']
slope_n_Mcut = data['Slope']
scatter_n_Mcut = data['Scatter']

errnorm_n_Mcut =  general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut =  general_functions.calculate_asymm_err(scatter_n_Mcut)


plt.errorbar(Lx_c,R_c,yerr=err_r_c,xerr=err_Lx_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
plt.errorbar(Lx_n,R_n,yerr=err_r_n,xerr = err_Lx_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(Lx_n)})')

sns.set_context('paper')
Lx_linspace = np.linspace(0.0001,125,100)
z_c = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(Lx_linspace,z_c,label='Best fit CC',color='black')
plt.plot(Lx_linspace,z_n,label='Best fit NCC',color='blue')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.legend(loc = 'lower right')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.title(r'$R-L_{\mathrm{X}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Lx_ccVncc_bestfit_Mcut.png',dpi=300,bbox_inches='tight')
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

general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=1,label=r'CC clusters', edgecolor='blue', lw = 1)
general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=3, edgecolor='blue', lw = 1)
plt.scatter(Slope_c_Mcut,Norm_c_Mcut,color = 'blue')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=1,label=r'NCC clusters', edgecolor='red', lw = 1)
general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=3, edgecolor='red', lw = 1)
plt.scatter(Slope_n_Mcut,Norm_n_Mcut,color = 'red')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
    
plt.xlim(-0.15,0.45)
plt.ylim(0.15,1.9)
plt.legend(prop = {'size' : 8}, loc='upper right')
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title(r'$R-L_{\mathrm{X}}$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Lx_ccVncc_contours.png', dpi = 300,bbox_inches='tight')

plt.show()





# =============================================================================
# z_n = ycept_n + Slope_n * log_Lx_n
# plt.errorbar( Z_n,z_n-log_r_n, yerr=sigma_r_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green')
# plt.axhline(0)
# #plt.xlim(-1.5,1.2)
# plt.ylim(2.5,-2.5)
# plt.show()
# 
# =============================================================================

# 2 keV cut
# =============================================================================
# c_new = CC_clusters[CC_clusters['Lx'] > 0.158]
# Z_c = (c_new['z'])
# E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
# R_c = c_new['R kpc']
# R_min_c = c_new['Rmin kpc']
# R_max_c = c_new['Rmax kpc']
# R_new_c = (R_c/250) * (E_c**(0.01))
# log_r_c = np.log10(R_new_c)
# sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))
# err_r_c = [(R_c - R_min_c)/250, (R_max_c-R_c)/250]
# 
# 
# Lx_c = c_new['Lx']
# log_Lx_c = np.log10(Lx_c)
# sigma_Lx_c = 0.4343*c_new['eL']/100
# 
# err_Lx_c = c_new['eL']*Lx_c/100
# 
# ycept_c,Norm_c,Slope_c,Scatter_c = general_functions.calculate_bestfit(log_Lx_c,sigma_Lx_c,log_r_c,sigma_r_c)
# 
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = c_new.sample(n = len(c_new), replace = True)
# # 
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(0.01))
# #     log_r = np.log10(R_new)
# #     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# # 
# # 
# #     Lx =random_clusters['Lx']
# #     log_Lx= np.log10(Lx)
# #     sigma_Lx = 0.4343*random_clusters['eL']/100
# #     
# # 
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# # 
# # 
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Lx_CC(cut)_BCES.csv')
# # 
# # =============================================================================
# 
# 
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_CC(cut)_BCES.csv')
# norm_c = data['Normalization']
# slope_c = data['Slope']
# scatter_c = data['Scatter']
# 
# errnorm_c = general_functions.calculate_asymm_err(norm_c)
# errslope_c = general_functions.calculate_asymm_err(slope_c)
# errscatter_c =  general_functions.calculate_asymm_err(scatter_c)
# 
# 
# # # NCC clusters
# 
# 
# n_new = NCC_clusters[NCC_clusters['Lx'] > 0.158]
# 
# Z_n = (n_new['z'])
# E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
# R_n = n_new['R kpc']
# R_min_n = n_new['Rmin kpc']
# R_max_n = n_new['Rmax kpc']
# R_new_n = (R_n/250) * (E_n**(0.01))
# log_r_n = np.log10(R_new_n)
# sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))
# err_r_n = [(R_n - R_min_n)/250, (R_max_n-R_n)/250]
# 
# 
# Lx_n = n_new['Lx']
# log_Lx_n = np.log10(Lx_n)
# sigma_Lx_n = 0.4343*n_new['eL']/100
# 
# err_Lx_n = n_new['eL']*Lx_n/100
# 
# ycept_n,Norm_n,Slope_n,Scatter_n = general_functions.calculate_bestfit(log_Lx_n,sigma_Lx_n,log_r_n,sigma_r_n)
# 
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     
# #     random_clusters = n_new.sample(n = len(n_new), replace = True)
# # 
# #     Z = (random_clusters['z'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(0.01))
# #     log_r = np.log10(R_new)
# #     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# # 
# # 
# #     Lx =random_clusters['Lx']
# #     log_Lx= np.log10(Lx)
# #     sigma_Lx = 0.4343*random_clusters['eL']/100
# #     
# # 
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# # 
# # 
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Lx_NCC(cut)_BCES.csv')
# # 
# # 
# # =============================================================================
# 
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_NCC(cut)_BCES.csv')
# norm_n = data['Normalization']
# slope_n = data['Slope']
# scatter_n = data['Scatter']
# 
# errnorm_n =  general_functions.calculate_asymm_err(norm_n)
# errslope_n = general_functions.calculate_asymm_err(slope_n)
# errscatter_n =  general_functions.calculate_asymm_err(scatter_n)
# 
# # =============================================================================
# # plt.errorbar(log_Lx_c,log_r_c,yerr=sigma_r_c,xerr=sigma_Lx_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
# # plt.errorbar(log_Lx_n,log_r_n,yerr=sigma_r_n,xerr=sigma_Lx_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_r_n)})')
# # 
# # #z = best_a + best_b*log_Lx
# # z_c = ycept_c + log_Lx_c*Slope_c
# # z_n = ycept_n + log_Lx_n*Slope_n
# # 
# # #plt.axvline(np.log10(0.15),color='Black')
# # 
# # plt.plot(log_Lx_c,z_c,label='Best fit CC',color='black')
# # plt.plot(log_Lx_n,z_n,label='Best fit NCC',color='blue')
# # 
# # #plt.xlim(1.5,25)
# # #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# # plt.xlabel('$log_{10}(L_{X}/10^{44}$ [erg/s])')
# # plt.ylabel(' $log_{10}(R*E(z)^{0.01}$ / 250 [kpc])')
# # plt.title('$R-L_{X}$ best fit  ')
# # plt.legend(loc='best')
# # #plt.savefig('R-Lx_best_fit-CCvNCC().png',dpi=300)
# # plt.show()
# # =============================================================================
# 
# 
# =============================================================================



# =============================================================================
# R_c = CC_clusters['R kpc']
# R_n = NCC_clusters['R kpc']
# 
# R_c = CC_clusters['Lx']
# R_n = NCC_clusters['Lx']
# 
# bins_c, cdf_c = general_functions.calculate_cdf(R_c, 20)
# bins_n, cdf_n = general_functions.calculate_cdf(R_n, 20)
# plt.plot(bins_c[1:], cdf_c,label = f'CC ({len(R_c)})')
# plt.plot(bins_n[1:], cdf_n, label = f'NCC ({len(R_n)})')
# plt.xscale('log')
# plt.xlabel('Lx [erg/s]')
# plt.ylabel('CDF')
# plt.title('CDF for Lx')
# plt.legend(loc='best')
# plt.show()
# general_functions.calculate_ks_stat(R_c, R_n)
# 
# 
# C_c = CC_clusters['c']
# err_c_c = CC_clusters['e_c']
# z_c = ycept_c + Slope_c* log_Lx_c
# z_n = ycept_n + Slope_n* log_Lx_n
# 
# =============================================================================
# =============================================================================
# C_n = NCC_clusters['c']
# err_c_n = NCC_clusters['e_c']
# plt.errorbar(C_c,z_c-log_r_c, yerr = sigma_r_c, xerr= err_c_c ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(z_c)})')
# plt.errorbar(C_n,z_n-log_r_n, yerr = sigma_r_n, xerr= err_c_n ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(z_n)})')
# plt.ylim(-1,1)
# plt.xlim(-0.1, 0.8)
# plt.xlabel('Concentration')
# plt.ylabel('$\Delta log_{10}R$')
# plt.title('$R-L_{X}$ residuals')
# plt.axhline(0, color = 'black')
# plt.axvline(0.18, color= 'blue', ls= '--', label='Threshold')
# plt.legend(loc = 'best')
# 
# plt.show()
# =============================================================================
















































###########################################################################
                            # New scaling relation after adding C
##############################################################################

r = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Half_radii_final_eeHIF_mass.csv')
r.rename({'# Name':'Cluster'},axis=1,inplace=True)
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


#For Rmin
theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
sigma_r = 0.4343 * (Rmax_kpc - Rmin_kpc)/(2*R_kpc)
rl['R kpc'] = R_kpc
rl['Rmin kpc'] = Rmin_kpc
rl['Rmax kpc'] = Rmax_kpc


thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rl_all = pd.merge(rl, thesis_table, left_on = rl['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how ='inner')
g = rl_all.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')



c_new = CC_clusters
Z_c = (c_new['z'])
E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
R_c = c_new['R kpc']
R_min_c = c_new['Rmin kpc']
R_max_c = c_new['Rmax kpc']
R_new_c = (R_c/250) * (E_c**(-0.68))
log_r_c = np.log10(R_new_c)
sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))
err_r_c = [(R_c - R_min_c)/250, (R_max_c-R_c)/250]

Lx_c = c_new['Lx']
log_Lx_c = np.log10(Lx_c)
sigma_Lx_c = 0.4343*c_new['eL']/100
err_Lx_c = c_new['eL']*Lx_c/100

c_c = c_new['c']/np.median(c_new['c'])
e_c_c = c_new['e_c']
log_c_c = np.log10(c_c)
sigma_c_c = 0.4343 * e_c_c/c_c
cov = np.cov(sigma_r_c,sigma_c_c)
yarray = log_r_c - (-0.44*log_c_c)
xarray = log_Lx_c
yerr = np.sqrt( (sigma_r_c)**2 + (-0.44*sigma_c_c)**2  )
xerr = sigma_Lx_c 
test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)

#Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
#     Z_c = (random_clusters['z'])
#     E_c = (omega_m*(1+Z_c)**3 + omega_lambda)**0.5
#     R_c = random_clusters['R kpc']
#     R_min_c = random_clusters['Rmin kpc']
#     R_max_c = random_clusters['Rmax kpc']
#     R_new_c = (R_c/250) * (E_c**(-0.68))
#     log_r_c = np.log10(R_new_c)
#     sigma_r_c = 0.4343 * ((R_max_c-R_min_c)/(2*R_c))
#     err_r_c = [(R_c - R_min_c)/250, (R_max_c-R_c)/250]
#     
#     Lx_c = random_clusters['Lx']
#     log_Lx_c = np.log10(Lx_c)
#     sigma_Lx_c = 0.4343*random_clusters['eL']/100
#     err_Lx_c = random_clusters['eL']*Lx_c/100
#     
#     c_c = random_clusters['c']/np.median(random_clusters['c'])
#     e_c_c = random_clusters['e_c']
#     log_c_c = np.log10(c_c)
#     sigma_c_c = 0.4343 * e_c_c/c_c
#     cov = np.cov(sigma_r_c,sigma_c_c)
#     yarray = log_r_c - (-0.44*log_c_c)
#     xarray = log_Lx_c
#     yerr = np.sqrt( (sigma_r_c)**2 + (-0.44*sigma_c_c)**2  )
#     xerr = sigma_Lx_c 
#     test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
#     best_A.append(test_Norm_c)
#     best_B.append(test_Slope_c)
#     best_scatter.append(test_Scatter_c)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_CC(with_C)_BCES.csv')  
# =============================================================================

# For NCC clusters
n_new = NCC_clusters
Z_n = (n_new['z'])
E_n = (omega_m*(1+Z_n)**3 + omega_lambda)**0.5
R_n = n_new['R kpc']
R_min_n = n_new['Rmin kpc']
R_max_n = n_new['Rmax kpc']
R_new_n = (R_n/250) * (E_n**(-0.68))
log_r_n = np.log10(R_new_n)
sigma_r_n = 0.4343 * ((R_max_n-R_min_n)/(2*R_n))
err_r_n = [(R_n - R_min_n)/250, (R_max_n-R_n)/250]

Lx_n = n_new['Lx']
log_Lx_n = np.log10(Lx_n)
sigma_Lx_n = 0.4343*n_new['eL']/100
err_Lx_n = n_new['eL']*Lx_n/100

c_n = n_new['c']/np.median(n_new['c'])
e_c_n = n_new['e_c']
log_c_n = np.log10(c_n)
sigma_c_n = 0.4343 * e_c_n/c_n
cov = np.cov(sigma_r_n,sigma_c_n)
yarray = log_r_n + 0.440*log_c_n
xarray = log_Lx_n
yerr = np.sqrt( (sigma_r_n)**2 + (-0.44*sigma_c_n)**2  )
xerr = sigma_Lx_n 
test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(-0.68))
#     log_r = np.log10(R_new)
#     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
#     err_r = [(R - R_min)/250, (R_max-R)/250]
#     
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL']/100
#     err_Lx = random_clusters['eL']*Lx_n/100
#     
#     c = random_clusters['c']/np.median(random_clusters['c'])
#     e_c = random_clusters['e_c']
#     log_c = np.log10(c)
#     sigma_c = 0.4343 * e_c/c
#     cov = np.cov(sigma_r,sigma_c)
#     yarray = log_r + 0.440*log_c
#     xarray = log_Lx
#     yerr = np.sqrt( (sigma_r)**2 + (-0.44*sigma_c)**2)
#     xerr = sigma_Lx
#     test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
#     best_A.append(test_Norm)
#     best_B.append(test_Slope)
#     best_scatter.append(test_Scatter)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_NCC(with_C)_BCES.csv')  
# =============================================================================
data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_CC(with_C)_BCES.csv')
norm_c = data_c['Normalization']
slope_c = data_c['Slope']
scatter_c = data_c['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c = general_functions.calculate_asymm_err(scatter_c)

data_n = pd.read_csv("/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_NCC(with_C)_BCES.csv")
norm_n = data_n['Normalization']
slope_n = data_n['Slope']
scatter_n = data_n['Scatter']

errnorm_n = general_functions.calculate_asymm_err(norm_n)
errslope_n = general_functions.calculate_asymm_err(slope_n)
errscatter_n = general_functions.calculate_asymm_err(scatter_n)

print(f'Normalization sigma after including C: {general_functions.calculate_sigma_dev(test_Norm_c,errnorm_c,test_Norm_n,errnorm_n)}')
print(f'Slope sigma after including C: {general_functions.calculate_sigma_dev(test_Slope_c,errslope_c,test_Slope_n,errslope_n)}')
print(f'Scatter sigma after including C: {general_functions.calculate_sigma_dev(test_Scatter_c,errscatter_c,test_Scatter_n,errscatter_n)}')


#z = best_a + best_b*log_Lx
# =============================================================================
# z_c = test_Ycept_c + log_Lx_c*test_Slope_c
# z_n = test_Ycept_n + log_Lx_n*test_Slope_n
# 
# #plt.axvline(np.log10(0.15),color='Black')
# 
# plt.plot(log_Lx_c,z_c,label='Best fit CC',color='black')
# plt.plot(log_Lx_n,z_n,label='Best fit NCC',color='blue')
# 
# #plt.xlim(1.5,25)
# #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# plt.xlabel('$log_{10}(L_{X}/10^{44}$ [erg/s])')
# plt.ylabel(' $log_{10}((R/c^{-0.44})*E(z)^{-0.68}$ / 250 [kpc])')
# plt.title('$R-L_{X}.c$ best fit  ')
# plt.legend(loc='best')
# #plt.savefig('R-Lx_best_fit-CCvNCC().png',dpi=300)
# plt.show()
# =============================================================================

err_c_y = (np.array(err_r_c)**2 + np.array(e_c_c)**2)**0.5
err_n_y = (np.array(err_r_n)**2 + np.array(e_c_n)**2)**0.5

plt.errorbar(Lx_c,R_new_c*(c_c)**-0.44,yerr=err_c_y,xerr=err_Lx_c,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(log_r_c)})')
plt.errorbar(Lx_n,R_new_n*(c_n)**-0.44,yerr=err_n_y,xerr=err_Lx_n,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(log_r_n)})')



#plt.axvline(np.log10(0.15),color='Black')
Lx_linspace = np.linspace(0.001,60,100)
z_c = test_Norm_c * Lx_linspace**test_Slope_c
z_n = test_Norm_n * Lx_linspace**test_Slope_n

plt.plot(Lx_linspace,z_c,label='Best fit',color='black')
plt.plot(Lx_linspace,z_n,label='Best fit NCC',color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.012, 45)
plt.ylim(0.05, 15)
rc('text',usetex=False)


plt.xlabel('$L_{\mathrm{X}}/10^{44}$ ($\mathrm{erg\,s^{-1}}$)')
plt.ylabel(r' $\frac{R}{c^{-0.44}}*E(z)^{-0.68}$  ($250\,\mathrm{kpc}$)')
plt.title(r'$R-L_{\mathrm{X}}$ best fit (including $C_{\mathrm{SB}}$)')

plt.legend(loc='upper left',prop={'size': 8})
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/R-Lx_ccVncc_with_c.png',dpi=300,bbox_inches='tight')
plt.show()