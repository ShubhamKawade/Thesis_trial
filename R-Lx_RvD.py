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
bestfit_Norm = bestfit_values['Norm_all'][6]
err_bestfit_Norm = bestfit_values['err_Norm_all'][6]
bestfit_Slope = bestfit_values['Slope_all'][6]
err_bestfit_Slope = bestfit_values['err_Slope_all'][6]
bestfit_Scatter = bestfit_values['Scatter_all'][6]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][6]


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
#for i in range(len(theta)):
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


offset = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_FINAL_ANISOTROPY_BCG_OFFSET.csv')
offset = general_functions.cleanup(offset)
rl_all = pd.merge(rl,offset, left_on = rl['Cluster'].str.casefold(), right_on = offset['Cluster'].str.casefold(), how ='inner')
r_clusters = rl_all[rl_all['BCG_offset_R500'] < 0.01]
d_clusters = rl_all[rl_all['BCG_offset_R500'] > 0.08 ]
 

Z_r = (r_clusters['z_x'])
E_r = (omega_m*(1+Z_r)**3 + omega_lambda)**0.5
R_r = r_clusters['R kpc']
R_min_r = r_clusters['Rmin kpc']
R_max_r = r_clusters['Rmax kpc']
R_new_r = (R_r/250) * (E_r**(-0.68))
log_r_r = np.log10(R_new_r)
sigma_r_r = 0.4343 * ((R_max_r-R_min_r)/(2*R_r))
err_r_r = [R_r - R_min_r, R_max_r-R_r]

Lx_r = r_clusters['Lx']
log_Lx_r = np.log10(Lx_r)
sigma_Lx_r = 0.4343*r_clusters['eL']/100
err_Lx_r = r_clusters['eL']*Lx_r/100
ycept_r,Norm_r,Slope_r,Scatter_r = general_functions.calculate_bestfit(log_Lx_r,sigma_Lx_r,log_r_r,sigma_r_r)

# Bootstrap : BCES

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = r_clusters.sample(n = len(r_clusters), replace = True)
# 
#     Z = (random_clusters['z_x'])
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
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_r_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_r_BCES.csv')
norm_r = data['Normalization']
slope_r = data['Slope']
scatter_r = data['Scatter']

errnorm_r =  general_functions.calculate_asymm_err(norm_r)
errslope_r = general_functions.calculate_asymm_err(slope_r)
errscatter_r =  general_functions.calculate_asymm_err(scatter_r)

# Disturbed clusters

Z_d = (d_clusters['z_x'])
E_d = (omega_m*(1+Z_d)**3 + omega_lambda)**0.5
R_d = d_clusters['R kpc']
R_min_d = d_clusters['Rmin kpc']
R_max_d = d_clusters['Rmax kpc']
R_new_d = (R_d/250) * (E_d**(-0.68))
log_r_d = np.log10(R_new_d)
sigma_r_d = 0.4343 * ((R_max_d-R_min_d)/(2*R_d))
err_r_d = [R_d-R_min_d, R_max_d - R_d]

Lx_d = d_clusters['Lx']
log_Lx_d = np.log10(Lx_d)
sigma_Lx_d = 0.4343*d_clusters['eL']/100
err_Lx_d = d_clusters['eL']*Lx_d/100
ycept_d,Norm_d,Slope_d,Scatter_d = general_functions.calculate_bestfit(log_Lx_d,sigma_Lx_d,log_r_d,sigma_r_d)


# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
# 
#     Z = (random_clusters['z_x'])
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
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_d_BCES.csv')
# =============================================================================


plt.errorbar(Lx_r,R_r,yerr=err_r_r,xerr=err_Lx_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'relaxed Clusters ({len(log_r_r)})')
plt.errorbar(Lx_d,R_d,yerr=err_r_d,xerr = err_Lx_d,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'disturbed Clusters ({len(Lx_d)})')

sns.set_context('paper')
Lx_linspace = np.linspace(0.0001,125,100)
z_r = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_r, Slope_r)
z_d = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_d, Slope_d)

plt.plot(Lx_linspace,z_r,label='Best fit relaxed',color='black')
plt.plot(Lx_linspace,z_d,label='Best fit disturbed',color='blue')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.title(r'$R-L_{\mathrm{X}}$ best fit')
plt.legend(loc = 'lower right')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.title('$R-L_{X}$ best fit')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/R-Lx_rVd_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_d_BCES.csv')
norm_d = data['Normalization']
slope_d = data['Slope']
scatter_d = data['Scatter']

errnorm_d =  general_functions.calculate_asymm_err(norm_d)
errslope_d = general_functions.calculate_asymm_err(slope_d)
errscatter_d = general_functions.calculate_asymm_err(scatter_d)


print('Re best fits:')
print(f'Normalization : {np.round(Norm_r,3)} +/- {np.round(errnorm_r,3)}')
print(f'Slope : {np.round(Slope_r,3)} +/- {np.round(errslope_r,3)}')
print(f'Scatter: {np.round(Scatter_r,3)} +/- {np.round(errscatter_r,3)}')

print('Di best fits:')

print(f'Normalization : {np.round(Norm_d,3)} +/- {np.round(errnorm_d,3)}')
print(f'Slope : {np.round(Slope_d,3)} +/- {np.round(errslope_d,3)}')
print(f'Scatter: {np.round(Scatter_d,3)} +/- {np.round(errscatter_d,3)}')


print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r,errnorm_r,Norm_d,errnorm_d)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r,Slope_d,errslope_d)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r,errscatter_r,Scatter_d,errscatter_d)}')


print(general_functions.percent_diff(Norm_r,errnorm_r,Norm_d,errnorm_d,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_r,errslope_r,Slope_d,errslope_d,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_r,errscatter_r,Scatter_d,errscatter_d,bestfit_Scatter, err_bestfit_Scatter))

sns.set_context('paper')
fig, ax_plot = plt.subplots()
#ax.scatter(slope_c,norm_c)
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=1,label=r'Relaxed contours', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_r,Norm_r,color = 'green', label='relaxed bestfit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=1,label=r'Disturbed contours', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_d,Norm_d,color = 'darkorange', label = 'disturbed best fit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

plt.xlim(-0.1,0.45)
plt.ylim(0.2,1.8)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-L_{X}$ : 1$\sigma$ & 3$\sigma$ contours for relaxed-disturbed')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Lx_relaxed_contours_full_sample.png' ,dpi=300, bbox_inches="tight")
plt.show()

##########################################################


   # Cutting galaxy groups based on mass


##########################################################

bestfit_Norm_clusters = bestfit_values['Norm_clusters'][6]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][6]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][6]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][6]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][6]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][6]


r_clusters = general_functions.removing_galaxy_groups(r_clusters)
Z_r = (r_clusters['z_x'])
E_r = (omega_m*(1+Z_r)**3 + omega_lambda)**0.5
R_r = r_clusters['R kpc']
R_min_r = r_clusters['Rmin kpc']
R_max_r = r_clusters['Rmax kpc']
R_new_r = (R_r/250) * (E_r**(-0.68))
log_r_r = np.log10(R_new_r)
sigma_r_r = 0.4343 * ((R_max_r-R_min_r)/(2*R_r))
err_r_r = [(R_r - R_min_r)/250, (R_max_r-R_r)/250]

Lx_r = r_clusters['Lx']
log_Lx_r = np.log10(Lx_r)
sigma_Lx_r = 0.4343*r_clusters['eL']/100
err_Lx_r = r_clusters['eL']*Lx_r/100
ycept_r_Mcut,Norm_r_Mcut,Slope_r_Mcut,Scatter_r_Mcut = general_functions.calculate_bestfit(log_Lx_r,sigma_Lx_r,log_r_r,sigma_r_r)

# Bootstrap : BCES

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = r_clusters.sample(n = len(r_clusters), replace = True)
# 
#     Z = (random_clusters['z_x'])
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
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_r(Mcut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_r(Mcut)_BCES.csv')
norm_r_Mcut = data['Normalization']
slope_r_Mcut = data['Slope']
scatter_r_Mcut = data['Scatter']

errnorm_r_Mcut =  general_functions.calculate_asymm_err(norm_r_Mcut)
errslope_r_Mcut = general_functions.calculate_asymm_err(slope_r_Mcut)
errscatter_r_Mcut =  general_functions.calculate_asymm_err(scatter_r_Mcut)

# Disturbed clusters
d_clusters = general_functions.removing_galaxy_groups(d_clusters)
Z_d = (d_clusters['z_x'])
E_d = (omega_m*(1+Z_d)**3 + omega_lambda)**0.5
R_d = d_clusters['R kpc']
R_min_d = d_clusters['Rmin kpc']
R_max_d = d_clusters['Rmax kpc']
R_new_d = (R_d/250) * (E_d**(-0.68))
log_r_d = np.log10(R_new_d)
sigma_r_d = 0.4343 * ((R_max_d-R_min_d)/(2*R_d))
err_r_d = [R_d-R_min_d, R_max_d - R_d]

Lx_d = d_clusters['Lx']
log_Lx_d = np.log10(Lx_d)
sigma_Lx_d = 0.4343*d_clusters['eL']/100
err_Lx_d = d_clusters['eL']*Lx_d/100
ycept_d_Mcut,Norm_d_Mcut,Slope_d_Mcut,Scatter_d_Mcut = general_functions.calculate_bestfit(log_Lx_d,sigma_Lx_d,log_r_d,sigma_r_d)


# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# 
# for j in range(0,10000):
#     
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
# 
#     Z = (random_clusters['z_x'])
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
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-Lx_d(Mcut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_d(Mcut)_BCES.csv')
norm_d_Mcut = data['Normalization']
slope_d_Mcut = data['Slope']
scatter_d_Mcut = data['Scatter']
errnorm_d_Mcut =  general_functions.calculate_asymm_err(norm_d_Mcut)
errslope_d_Mcut = general_functions.calculate_asymm_err(slope_d_Mcut)
errscatter_d_Mcut = general_functions.calculate_asymm_err(scatter_d_Mcut)

sns.set_context('paper')

plt.errorbar(Lx_r,R_r,yerr=err_r_r,xerr=err_Lx_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'relaxed Clusters ({len(log_r_r)})')
plt.errorbar(Lx_d,R_d,yerr=err_r_d,xerr = err_Lx_d,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'disturbed Clusters ({len(Lx_d)})')
Lx_linspace = np.linspace(0.0001,125,100)
z_r = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_r_Mcut, Slope_r_Mcut)
z_d = general_functions.plot_bestfit(Lx_linspace, 1, 250, ycept_d_Mcut, Slope_d_Mcut)

plt.plot(Lx_linspace,z_r,label='Best fit relaxed',color='black')
plt.plot(Lx_linspace,z_d,label='Best fit disturbed',color='blue')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel('$L_{\mathrm{X}}$ ($*\,10^{44}$ $\mathrm{erg\,s^{-1}}$)')
plt.ylabel(' $R*E(z)^{-0.68}$ (kpc)')
plt.xlim(0.002,80)
plt.ylim(10,2400)
plt.title(r'$R-L_{\mathrm{X}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/R-Lx_rVd(Mcut)_bestfit.png',dpi=300,bbox_inches="tight")

plt.show()

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r_Mcut,errnorm_r_Mcut,Norm_d_Mcut,errnorm_d_Mcut)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r_Mcut,Slope_d_Mcut,errslope_d_Mcut)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r_Mcut,errscatter_r_Mcut,Scatter_d_Mcut,errscatter_d_Mcut)}')


print(general_functions.percent_diff(Norm_r_Mcut,errnorm_r_Mcut,Norm_d_Mcut,errnorm_d_Mcut,bestfit_Norm_clusters, err_bestfit_Norm_clusters))
print(general_functions.percent_diff(Slope_r_Mcut,errslope_r_Mcut,Slope_d_Mcut,errslope_d_Mcut,bestfit_Slope_clusters, err_bestfit_Slope_clusters))
print(general_functions.percent_diff(Scatter_r_Mcut,errscatter_r_Mcut,Scatter_d_Mcut,errscatter_d_Mcut,bestfit_Scatter_clusters, err_bestfit_Scatter_clusters))

sns.set_context('paper')
fig, ax_plot = plt.subplots()
#ax.scatter(slope_c,norm_c)
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=1,label=r'Relaxed (clusters+groups)', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_r,Norm_r,color = 'green')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=1,label=r'Disturbed (clusters+groups)', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_d,Norm_d,color = 'darkorange')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_r_Mcut, norm_r_Mcut, Slope_r_Mcut, Norm_r_Mcut, ax_plot, n_std=1,label=r'Relaxed (clusters)', edgecolor='blue', lw = 1)
general_functions.confidence_ellipse(slope_r_Mcut, norm_r_Mcut, Slope_r_Mcut, Norm_r_Mcut, ax_plot, n_std=3, edgecolor='blue', lw = 1)
plt.scatter(Slope_r_Mcut,Norm_r_Mcut,color = 'blue')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_d_Mcut, norm_d_Mcut, Slope_d_Mcut, Norm_d_Mcut, ax_plot, n_std=1,label=r'Disturbed  (clusters)', edgecolor='red', lw = 1)
general_functions.confidence_ellipse(slope_d_Mcut, norm_d_Mcut, Slope_d_Mcut, Norm_d_Mcut, ax_plot, n_std=3, edgecolor='red', lw = 1)
plt.scatter(Slope_d_Mcut,Norm_d_Mcut,color = 'red')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
    
plt.xlim(-0.3,0.8)
plt.ylim(0,2.0)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$R-L_{\mathrm{X}}$ : 1$\sigma$ & 3$\sigma$ contours for relaxed-disturbed')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/R-Lx_relaxed_contours.png' ,dpi=300, bbox_inches="tight")

plt.show()
# # Lx cut
# relaxed clusters
# =============================================================================
# r_new = r_clusters[r_clusters['Lx'] > 0.158]
# Z_r = (r_new['z_x'])
# E_r = (omega_m*(1+Z_r)**3 + omega_lambda)**0.5
# R_r = r_new['R kpc']
# R_min_r = r_new['Rmin kpc']
# R_max_r = r_new['Rmax kpc']
# R_new_r = (R_r/250) * (E_r**(0.02))
# log_r_r = np.log10(R_new_r)
# sigma_r_r = 0.4343 * ((R_max_r-R_min_r)/(2*R_r))
# err_r_r = [(R_r - R_min_r)/250, (R_max_r-R_r)/250]
# 
# Lx_r = r_new['Lx']
# log_Lx_r = np.log10(Lx_r)
# sigma_Lx_r = 0.4343*r_new['eL']/100
# err_Lx_r = r_new['eL']*Lx_r/100
# ycept_r,Norm_r,Slope_r,Scatter_r = general_functions.calculate_bestfit(log_Lx_r,sigma_Lx_r,log_r_r,sigma_r_r)
# 
# # Bootstrap
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # 
# # for j in range(0,10000):
# #     
# #     random_clusters = r_new.sample(n = len(r_new), replace = True)
# # 
# #     Z = (random_clusters['z_x'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(0.02))
# #     log_r = np.log10(R_new)
# #     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# # 
# # 
# #     Lx =random_clusters['Lx']
# #     log_Lx= np.log10(Lx)
# #     sigma_Lx = 0.4343*random_clusters['eL']/100
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# # 
# # 
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Lx_r(cut)_BCES.csv')
# # 
# # =============================================================================
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_r(cut)_BCES.csv')
# norm_r = data['Normalization']
# slope_r = data['Slope']
# scatter_r = data['Scatter']
# 
# errnorm_r =  general_functions.calculate_asymm_err(norm_r)
# errslope_r = general_functions.calculate_asymm_err(slope_r)
# errscatter_r =  general_functions.calculate_asymm_err(scatter_r)
# 
# # Disturbed clusters
# d_new = d_clusters[d_clusters['Lx'] > 0.158]
# 
# Z_d = (d_new['z_x'])
# E_d = (omega_m*(1+Z_d)**3 + omega_lambda)**0.5
# R_d = d_new['R kpc']
# R_min_d = d_new['Rmin kpc']
# R_max_d = d_new['Rmax kpc']
# R_new_d = (R_d/250) * (E_d**(0.02))
# log_r_d = np.log10(R_new_d)
# sigma_r_d = 0.4343 * ((R_max_d-R_min_d)/(2*R_d))
# err_r_d = [(R_d-R_min_d)/250, (R_max_d - R_d)/250]
# 
# Lx_d = d_new['Lx']
# log_Lx_d = np.log10(Lx_d)
# sigma_Lx_d = 0.4343*d_new['eL']/100
# err_Lx_d = d_new['eL']*Lx_d/100
# 
# ycept_d,Norm_d,Slope_d,Scatter_d = general_functions.calculate_bestfit(log_Lx_d,sigma_Lx_d,log_r_d,sigma_r_d)
# 
# 
# 
# 
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # 
# # for j in range(0,10000):
# #     
# #     random_clusters = d_new.sample(n = len(d_new), replace = True)
# # 
# #     Z = (random_clusters['z_x'])
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     R = random_clusters['R kpc']
# #     R_min = random_clusters['Rmin kpc']
# #     R_max = random_clusters['Rmax kpc']
# #     R_new = (R/250) * (E**(0.02))
# #     log_r = np.log10(R_new)
# #     sigma_r = 0.4343 * ((R_max-R_min)/(2*R))
# # 
# # 
# #     Lx =random_clusters['Lx']
# #     log_Lx= np.log10(Lx)
# #     sigma_Lx = 0.4343*random_clusters['eL']/100
# #     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lx,sigma_Lx,log_r,sigma_r)
# # 
# # 
# #     best_A.append(Norm)
# #     best_B.append(Slope)
# #     best_scatter.append(Scatter)
# #     
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# # bestfit_bootstrap.to_csv('R-Lx_d(cut)_BCES.csv')
# # 
# # 
# # =============================================================================
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-Lx_d(cut)_BCES.csv')
# norm_d = data['Normalization']
# slope_d = data['Slope']
# scatter_d = data['Scatter']
# 
# errnorm_d =  general_functions.calculate_asymm_err(norm_d)
# errslope_d = general_functions.calculate_asymm_err(slope_d)
# errscatter_d = general_functions.calculate_asymm_err(scatter_d)
# 
# 
# plt.errorbar(Lx_r,R_new_r,yerr=err_r_r,xerr=err_Lx_r,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'relaxed Clusters ({len(log_r_r)})')
# plt.errorbar(Lx_d,R_new_d,yerr=err_r_d,xerr = err_Lx_d,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'disturbed Clusters ({len(Lx_d)})')
# z_r = Norm_r * Lx_r**Slope_r
# z_d = Norm_d * Lx_d**Slope_d
# plt.axvline(0.15,color='Black')
# 
# plt.plot(Lx_r,z_r,label='Best fit relaxed',color='black')
# plt.plot(Lx_d,z_d,label='Best fit disturbed',color='blue')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(0.2,4)
# #plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
# plt.xlabel('$L_{X}/10^{44}$ [erg/s]')
# plt.ylabel(' $R*E(z)^{-0.83}$ / 250 [kpc]')
# plt.title(r'$R-L_{X}$ best fit')
# plt.legend(loc = 'best')
# #plt.savefig('R-Lx_best_fit-rVd(cut).png',dpi=300)
# plt.show()
# 
# 
# print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r,errnorm_r,Norm_d,errnorm_d)}')
# print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r,Slope_d,errslope_d)}')
# print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r,errscatter_r,Scatter_d,errscatter_d)}')
# 
# 
# print(general_functions.percent_diff(Norm_r,errnorm_r,Norm_d,errnorm_d))
# print(general_functions.percent_diff(Slope_r,errslope_r,Slope_d,errslope_d))
# print(general_functions.percent_diff(Scatter_r,errscatter_r,Scatter_d,errscatter_d))
# 
# 
# R_r = r_clusters['R kpc']
# R_d = d_clusters['R kpc']
# bins_r, cdf_r = general_functions.calculate_cdf(R_r, 20)
# bins_d, cdf_d = general_functions.calculate_cdf(R_d, 20)
# plt.plot(bins_r[1:], cdf_r,label = f'relaxed ({len(R_r)})')
# plt.plot(bins_d[1:], cdf_d, label = f'disturbed ({len(R_d)})')
# plt.xlabel('R [kpc]')
# plt.ylabel('CDF')
# plt.title('CDF for R (0.15e44 erg/s Lx cut)')
# plt.legend(loc='best')
# #plt.savefig('CDF_R(cut)_rVd.png',dpi = 300)
# 
# general_functions.calculate_ks_stat(R_r, R_d)
# 
# 
# 
# =============================================================================
