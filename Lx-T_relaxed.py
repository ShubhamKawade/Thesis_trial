#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
import seaborn as sns

#Importing the best fit values of the scaling relation fit

bestfit_values = pd.read_csv('/home/schubham/Thesis/Thesis/Data/best_fit_parameters.csv')
bestfit_Norm = bestfit_values['Norm_all'][0]
err_bestfit_Norm = bestfit_values['err_Norm_all'][0]
bestfit_Slope = bestfit_values['Slope_all'][0]
err_bestfit_Slope = bestfit_values['err_Slope_all'][0]
bestfit_Scatter = bestfit_values['Scatter_all'][0]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][0]

master_file = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')
master_file = master_file.rename({'#Cluster':'Cluster'},axis=1)
master_file = general_functions.cleanup(master_file)

offset = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_FINAL_ANISOTROPY_BCG_OFFSET.csv')
offset = general_functions.cleanup(offset)
cluster_total = pd.merge(master_file, offset, how='inner', left_on = master_file['Cluster'].str.casefold(), right_on = offset['Cluster'].str.casefold())
# relaxed clusters
r_clusters = cluster_total[cluster_total['BCG_offset_R500'] < 0.01 ]
d_clusters = cluster_total[cluster_total['BCG_offset_R500'] > 0.08 ]

len(r_clusters)
len(d_clusters)
len(offset)
offset['BCG_offset_R500']

Z = (r_clusters['z_x'])
##USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7
Lx = r_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_r = r_clusters['Lx(1e44)']/E
T_r = r_clusters['T(keV)_x']
T_new_r = T_r/4.5

log_Lx_r = np.log10(Lx_new_r)
log_T_r = np.log10(T_r)
log_T_new_r = np.log10(T_new_r)
sigma_Lx_r = 0.4343*r_clusters['eL(%)']/100
sigma_T_r = 0.4343*((r_clusters['Tmax_x']-r_clusters['Tmin_x'])/(2*T_r))
err_Lx_r = r_clusters['eL(%)']*Lx_new_r/100
err_T_r = [T_r- r_clusters['Tmin_x'], r_clusters['Tmax_x']-T_r]
ycept_r,Norm_r,Slope_r,Scatter_r = general_functions.calculate_bestfit(log_T_new_r,sigma_T_r,log_Lx_r,sigma_Lx_r)



#Bootstrap for relaxed #

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = r_clusters.sample(n = len(r_clusters), replace = True)
#     Z = (random_clusters['z_x'])
#     #   ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)_x']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax_x']-random_clusters['Tmin_x'])/(2*T))
#     ycept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_r_BCES.csv')  
# =============================================================================

# Disturbed clusters
Z = (d_clusters['z_x'])
##USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7

Lx = d_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_d = d_clusters['Lx(1e44)']/E
T_d = d_clusters['T(keV)_x']
T_new_d = T_d/4.5

log_Lx_d = np.log10(Lx_new_d)
log_T_d = np.log10(T_d)
log_T_new_d = np.log10(T_new_d)
sigma_Lx_d = 0.4343*d_clusters['eL(%)']/100
sigma_T_d = 0.4343*((d_clusters['Tmax_x']-d_clusters['Tmin_x'])/(2*T_d))
err_Lx_d = d_clusters['eL(%)']*Lx_new_d/100
err_T_d = [T_d - d_clusters['Tmin_x'], d_clusters['Tmax_x']-T_d]

ycept_d,Norm_d,Slope_d,Scatter_d = general_functions.calculate_bestfit(log_T_new_d,sigma_T_d,log_Lx_d,sigma_Lx_d)

#Bootstrap for disturbed #


# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
#     Z = (random_clusters['z_x'])
#     #   ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)_x']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax_x']-random_clusters['Tmin_x'])/(2*T))
#     ycept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_d_BCES.csv')  
# 
# =============================================================================


data_r = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_r_BCES.csv')
norm_r = data_r['Normalization']
slope_r = data_r['Slope']
scatter_r = data_r['Scatter']

errnorm_r =  general_functions.calculate_asymm_err(norm_r)
errslope_r = general_functions.calculate_asymm_err(slope_r)
errscatter_r = general_functions.calculate_asymm_err(scatter_r)



data_d = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_d_BCES.csv')
norm_d = data_d['Normalization']
slope_d = data_d['Slope']
scatter_d = data_d['Scatter']

errnorm_d =  general_functions.calculate_asymm_err(norm_d)
errslope_d = general_functions.calculate_asymm_err(slope_d)
errscatter_d =  general_functions.calculate_asymm_err(scatter_d)


print('Re best fits:')
print(f'Normalization : {np.round(Norm_r,3)} +/- {np.round(errnorm_r,3)}')
print(f'Slope : {np.round(Slope_r,3)} +/- {np.round(errslope_r,3)}')
print(f'Scatter: {np.round(Scatter_r,3)} +/- {np.round(errscatter_r,3)}')

print('Di best fits:')

print(f'Normalization : {np.round(Norm_d,3)} +/- {np.round(errnorm_d,3)}')
print(f'Slope : {np.round(Slope_d,3)} +/- {np.round(errslope_d,3)}')
print(f'Scatter: {np.round(Scatter_d,3)} +/- {np.round(errscatter_d,3)}')



sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_r = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_r, Slope_r)
z_d = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_d, Slope_d)

plt.plot(T_linspace,z_r,label='Best fit relaxed', color ='blue')
plt.plot(T_linspace,z_d,label='Best fit disturbed', color ='black')


plt.errorbar(T_r, Lx_new_r, xerr=err_T_r, color = 'green', yerr=err_Lx_r, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'relaxed clusters ({len(T_r)})')
plt.errorbar(T_d, Lx_new_d, xerr=err_T_d, color='red', yerr=err_Lx_d, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'disturbed clusters ({len(T_d)})')
#plt.fill_between(T,lcb,ucb, facecolor='red')
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg\,s^{-1}}$)')
plt.title('$L_{\mathrm{X}}-T$ best fit ')
plt.legend(loc = 'lower right')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/Lx-T_rVd_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r,errnorm_r,Norm_d,errnorm_d)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r,Slope_d,errslope_d)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r,errscatter_r,Scatter_d,errscatter_d)}')


print(general_functions.percent_diff(Norm_r,errnorm_r,Norm_d,errnorm_d,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_r,errslope_r,Slope_d,errslope_d,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_r,errscatter_r,Scatter_d,errscatter_d,bestfit_Scatter, err_bestfit_Scatter))

# =============================================================================
# sns.set_context('paper')
# fig, ax_plot = plt.subplots()
# #ax.scatter(slope_c,norm_c)
# general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=1,label=r'Relaxed contours', edgecolor='green', lw = 1)
# general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=3, edgecolor='green', lw = 1)
# plt.scatter(Slope_r,Norm_r,color = 'green', label='relaxed bestfit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
#      
# general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=1,label=r'Disturbed contours', edgecolor='darkorange', lw = 1)
# general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
# plt.scatter(Slope_d,Norm_d,color = 'darkorange', label = 'disturbed best fit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
# 
# 
# plt.xlim(1.3,3)
# plt.ylim(0.75,2.5)
# plt.legend(prop = {'size' : 8})
# plt.xlabel('Slope')
# plt.ylabel('Normalization')
# plt.title('$L_{X}-T$ : 1$\sigma$ & 3$\sigma$ contours for relaxed-disturbed')
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Lx-T_relaxed_contours_full_sample.png' ,dpi=300, bbox_inches="tight")
# plt.show()
# 
# =============================================================================

##########################################################


   # Cutting galaxy groups based on mass


##########################################################

bestfit_Norm_clusters = bestfit_values['Norm_clusters'][0]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][0]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][0]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][0]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][0]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][0]

# Relaxed clusters
r_clusters = general_functions.removing_galaxy_groups(r_clusters)
Z = (r_clusters['z_x'])
##USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7
Lx = r_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_r = r_clusters['Lx(1e44)']/E
T_r = r_clusters['T(keV)_x']
T_new_r = T_r/4.5

log_Lx_r = np.log10(Lx_new_r)
log_T_r = np.log10(T_r)
log_T_new_r = np.log10(T_new_r)
sigma_Lx_r = 0.4343*r_clusters['eL(%)']/100
sigma_T_r = 0.4343*((r_clusters['Tmax_x']-r_clusters['Tmin_x'])/(2*T_r))
err_Lx_r = r_clusters['eL(%)']*Lx_new_r/100
err_T_r = [T_r- r_clusters['Tmin_x'], r_clusters['Tmax_x']-T_r]
ycept_r_Mcut,Norm_r_Mcut,Slope_r_Mcut,Scatter_r_Mcut = general_functions.calculate_bestfit(log_T_new_r,sigma_T_r,log_Lx_r,sigma_Lx_r)



#Bootstrap for relaxed #

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = r_clusters.sample(n = len(r_clusters), replace = True)
#     Z = (random_clusters['z_x'])
#     #   ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)_x']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax_x']-random_clusters['Tmin_x'])/(2*T))
#     ycept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_r(Mcut)_BCES.csv')  
# =============================================================================

# Disturbed clusters
d_clusters = general_functions.removing_galaxy_groups(d_clusters)
Z = (d_clusters['z_x'])
Lx = d_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_d = d_clusters['Lx(1e44)']/E
T_d = d_clusters['T(keV)_x']
T_new_d = T_d/4.5

log_Lx_d = np.log10(Lx_new_d)
log_T_d = np.log10(T_d)
log_T_new_d = np.log10(T_new_d)
sigma_Lx_d = 0.4343*d_clusters['eL(%)']/100
sigma_T_d = 0.4343*((d_clusters['Tmax_x']-d_clusters['Tmin_x'])/(2*T_d))
err_Lx_d = d_clusters['eL(%)']*Lx_new_d/100
err_T_d = [T_d - d_clusters['Tmin_x'], d_clusters['Tmax_x']-T_d]

ycept_d_Mcut,Norm_d_Mcut,Slope_d_Mcut,Scatter_d_Mcut = general_functions.calculate_bestfit(log_T_new_d,sigma_T_d,log_Lx_d,sigma_Lx_d)

#Bootstrap for disturbed #


# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
#     Z = (random_clusters['z_x'])
#     #   ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)_x']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax_x']-random_clusters['Tmin_x'])/(2*T))
#     ycept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_d(Mcut)_BCES.csv')  
# 
# =============================================================================


data_r = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_r(Mcut)_BCES.csv')
norm_r_Mcut = data_r['Normalization']
slope_r_Mcut = data_r['Slope']
scatter_r_Mcut= data_r['Scatter']

errnorm_r_Mcut =  general_functions.calculate_asymm_err(norm_r_Mcut)
errslope_r_Mcut = general_functions.calculate_asymm_err(slope_r_Mcut)
errscatter_r_Mcut = general_functions.calculate_asymm_err(scatter_r_Mcut)



data_d = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_d(Mcut)_BCES.csv')
norm_d_Mcut = data_d['Normalization']
slope_d_Mcut = data_d['Slope']
scatter_d_Mcut = data_d['Scatter']

errnorm_d_Mcut =  general_functions.calculate_asymm_err(norm_d_Mcut)
errslope_d_Mcut= general_functions.calculate_asymm_err(slope_d_Mcut)
errscatter_d_Mcut =  general_functions.calculate_asymm_err(scatter_d_Mcut)

sns.set_context('paper')
plt.errorbar(T_r,Lx_new_r,xerr= err_T_r,color = 'green',yerr=err_Lx_r,ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'relaxed clusters ({len(T_r)})' )
plt.errorbar(T_d,Lx_new_d,xerr= err_T_d,color = 'red',yerr=err_Lx_d,ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'disturbed clusters ({len(T_d)})' )

sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_r = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_r_Mcut, Slope_r_Mcut)
z_d = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_d_Mcut, Slope_d_Mcut)

plt.plot(T_linspace,z_r,label='Best fit relaxed', color ='blue')
plt.plot(T_linspace,z_d,label='Best fit disturbed', color ='black')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg\,s^{-1}}$)')
plt.title(r'$L_{\mathrm{X}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}M_{\odot}$)')
plt.legend(loc='lower right')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/Lx-T_rVd(Mcut)_bestfit.png',dpi=300,bbox_inches="tight")

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
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=1,label=r'Relaxed (clusters)', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_r, norm_r, Slope_r, Norm_r, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_r,Norm_r,color = 'green')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=1,label=r'Disturbed (clusters)', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_d, norm_d, Slope_d, Norm_d, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_d,Norm_d,color = 'darkorange')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_r_Mcut, norm_r_Mcut, Slope_r_Mcut, Norm_r_Mcut, ax_plot, n_std=1,label=r'Relaxed (clusters+groups)', edgecolor='blue', lw = 1)
general_functions.confidence_ellipse(slope_r_Mcut, norm_r_Mcut, Slope_r_Mcut, Norm_r_Mcut, ax_plot, n_std=3, edgecolor='blue', lw = 1)
plt.scatter(Slope_r_Mcut,Norm_r_Mcut,color = 'blue')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_d_Mcut, norm_d_Mcut, Slope_d_Mcut, Norm_d_Mcut, ax_plot, n_std=1,label=r'Disturbed (clusters+groups)', edgecolor='red', lw = 1)
general_functions.confidence_ellipse(slope_d_Mcut, norm_d_Mcut, Slope_d_Mcut, Norm_d_Mcut, ax_plot, n_std=3, edgecolor='red', lw = 1)
plt.scatter(Slope_d_Mcut,Norm_d_Mcut,color = 'red')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
    
plt.xlim(0.5,3)
plt.ylim(0.75,3.2)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$L_{X}-T$ : 1$\sigma$ & 3$\sigma$ contours for relaxed-disturbed clusters')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Lx-T_relaxed_contours.png' ,dpi=300, bbox_inches="tight")
plt.show()


# ===========================================================================
# =============================================================================
# # 2 keV cut
# # Relaxed clusters
# 
# r_clusters = r_clusters[r_clusters['T(keV)'] > 2]
# Z = (r_clusters['z'])
# ##USING DENSITY PARAMETERS ACCORDING TO LCDM
# omega_m = 0.3
# omega_lambda = 0.7
# 
# Lx_r = r_clusters['Lx(1e44erg/s)']
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# Lx_new_r = Lx_r/E
# T_r = r_clusters['T(keV)']
# T_new_r = T_r/4.5
# 
# log_Lx_r = np.log10(Lx_new_r)
# log_T_r = np.log10(T_r)
# log_T_new_r = np.log10(T_new_r)
# sigma_Lx_r = 0.4343*r_clusters['e_L(%)']/100
# sigma_T_r = 0.4343*((r_clusters['Tmax']-r_clusters['Tmin'])/(2*T_r))
# err_Lx_r = r_clusters['e_L(%)']*Lx_new_r/100
# err_T_r = [T_r-r_clusters['Tmin'], r_clusters['Tmax']-T_r]
# ycept_r,Norm_r,Slope_r,Scatter_r = general_functions.calculate_bestfit(log_T_new_r,sigma_T_r,log_Lx_r,sigma_Lx_r)
# 
# 
# 
# 
# # best_A = []
# # best_B = []
# # best_scatter = []
# # #cluster_total_CC = cluster_total_CC.to_pandas()
# # for j in range(0,10000):
# #     random_clusters = r_clusters.sample(n = len(r_clusters), replace = True)
#     
# #     Z = (random_clusters['z'])
# #     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
# #     omega_m = 0.3
# #     omega_lambda = 0.7
# 
# #     Lx = random_clusters['Lx(1e44erg/s)']
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     Lx_new = random_clusters['Lx(1e44erg/s)']/E
# #     T = random_clusters['T(keV)']
# #     T_new = T/4.5
# 
# #     log_Lx_c = np.log10(Lx_new)
# #     log_T_c = np.log10(T)
# #     log_T_new_c = np.log10(T_new)
# #     sigma_Lx_c = 0.4343*random_clusters['e_L(%)']/100
# #     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
# #     yintercept,norm,slope,scatter = Functions.bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
# #     best_A.append(norm)
# #     best_B.append(slope)
# #     best_scatter.append(scatter)
# 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
#   
# # # saving the dataframe 
# # bestfit_bootstrap.to_csv('Lx-T_r(2kev)_BCES.csv')  
# # %time
# 
# 
# 
# # Disturbed clusters
# d_clusters = d_clusters[d_clusters['T(keV)'] > 2]
# Z = (d_clusters['z'])
# # USING DENSITY PARAMETERS ACCORDING TO LCDM
# omega_m = 0.3
# omega_lambda = 0.7
# 
# Lx_d = d_clusters['Lx(1e44erg/s)']
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# Lx_new_d = Lx_d/E
# T_d = d_clusters['T(keV)']
# T_new_d = T_d/4.5
# 
# log_Lx_d = np.log10(Lx_new_d)
# log_T_d = np.log10(T_d)
# log_T_new_d = np.log10(T_new_d)
# sigma_Lx_d = 0.4343*d_clusters['e_L(%)']/100
# sigma_T_d = 0.4343*((d_clusters['Tmax']-d_clusters['Tmin'])/(2*T_d))
# err_Lx_d = d_clusters['e_L(%)']*Lx_new_d/100
# err_T_d = [T_d-d_clusters['Tmin'], d_clusters['Tmax']-T_d]
# ycept_d,Norm_d,Slope_d,Scatter_d = general_functions.calculate_bestfit(log_T_new_d,sigma_T_d,log_Lx_d,sigma_Lx_d)
# 
# 
# #Bootstrap for disturbed #
# 
# # best_A = []
# # best_B = []
# # best_scatter = []
# # for j in range(0,10000):
# #     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
#     
# #     Z = (random_clusters['z'])
# #     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
# #     omega_m = 0.3
# #     omega_lambda = 0.7
# 
# #     Lx = random_clusters['Lx(1e44erg/s)']
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     Lx_new = random_clusters['Lx(1e44erg/s)']/E
# #     T = random_clusters['T(keV)']
# #     T_new = T/4.5
# 
# #     log_Lx_c = np.log10(Lx_new)
# #     log_T_c = np.log10(T)
# #     log_T_new_c = np.log10(T_new)
# #     sigma_Lx_c = 0.4343*random_clusters['e_L(%)']/100
# #     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
# #     yintercept,norm,slope,scatter = Functions.bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
# #     best_A.append(norm)
# #     best_B.append(slope)
# #     best_scatter.append(scatter)
# 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
#   
# # # saving the dataframe 
# # bestfit_bootstrap.to_csv('Lx-T_d(2kev)_BCES.csv')  
# # %time
# 
# data_r_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_r(2kev)_BCES.csv')
# norm_r = data_r_n['Normalization']
# slope_r = data_r_n['Slope']
# scatter_r = data_r_n['Scatter']
# 
# errnorm_r =  general_functions.calculate_asymm_err(norm_r)
# errslope_r = general_functions.calculate_asymm_err(slope_r)
# errscatter_r =  general_functions.calculate_asymm_err(scatter_r)
# 
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_d(2kev)_BCES.csv')
# norm_d = data['Normalization']
# slope_d = data['Slope']
# scatter_d = data['Scatter']
# 
# errnorm_d =  general_functions.calculate_asymm_err(norm_d)
# errslope_d = general_functions.calculate_asymm_err(slope_d)
# errscatter_d =  general_functions.calculate_asymm_err(scatter_d)
# 
# 
# sns.set_context('notebook')
# plt.errorbar(T_r,Lx_new_r,xerr= err_T_r,color = 'green',yerr=err_Lx_r,ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'relaxed clusters ({len(T_r)})' )
# plt.errorbar(T_d,Lx_new_d,xerr= err_T_d,color = 'red',yerr=err_Lx_d,ls='',fmt='.', capsize = 2,alpha= 1, elinewidth = 0.6, label = f'disturbed clusters ({len(T_d)})' )
# 
# z_r = Norm_r * (T_new_r)**Slope_r
# z_d = Norm_d * (T_new_d)**Slope_d
# 
# plt.plot(T_r,z_r, color = 'blue',label = 'relaxed bestfit')
# plt.plot(T_d,z_d,color = 'black', label = 'distubed bestfit')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('T keV')
# plt.ylabel('$L_{X}$/E(z) ($10^{44}$ erg/s)')
# plt.title('$L_{X}-T$ best fit (2keV cut)')
# plt.legend(bbox_to_anchor=[0.6,0.31])
# #plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r,errnorm_r,Norm_d,errnorm_d)}')
# print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r,Slope_d,errslope_d)}')
# print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r,errscatter_r,Scatter_d,errscatter_d)}')
# 
# bins_r, cdf_r = general_functions.calculate_cdf(Lx_r,10)
# bins_d, cdf_d = general_functions.calculate_cdf(Lx_d,10)
# 
# plt.plot(bins_r[1:],cdf_r,label = 'R')
# plt.plot(bins_d[1:],cdf_d, label = 'D')
# plt.xlabel('Lx*$10^{44}$ [erg/s]')
# plt.ylabel('CDF')
# plt.title('CDF comparison')
# plt.legend(loc = 'best')
# plt.show()
# 
# 
# print(general_functions.percent_diff(Norm_r,errnorm_r,Norm_d,errnorm_d))
# print(general_functions.percent_diff(Slope_r,errslope_r,Slope_d,errslope_d))
# print(general_functions.percent_diff(Scatter_r,errscatter_r,Scatter_d,errscatter_d))
# 
# # =============================================================================
# # T_r = r_clusters['T(keV)']
# # T_d = d_clusters['T(keV)']
# # bins_r, cdf_r = general_functions.calculate_cdf(T_r, 20)
# # bins_d, cdf_d = general_functions.calculate_cdf(T_d, 20)
# # plt.plot(bins_r[1:], cdf_r,label = f'relaxed ({len(T_r)})')
# # plt.plot(bins_d[1:], cdf_d, label = f'disturbed ({len(T_d)})')
# # plt.xlabel('T [keV]')
# # plt.ylabel('CDF')
# # plt.title('CDF for T')
# # plt.legend(loc='best')
# # #plt.savefig('CDF_T_rVd.png',dpi = 300)
# # 
# # general_functions.calculate_ks_stat(T_r, T_d)
# # 
# # =============================================================================
# 
# =============================================================================

