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
master_file = general_functions.cleanup(master_file)

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv', sep =',')
#thesis_table = thesis_table.rename({'#Cluster':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)

# =============================================================================
# # To check if all the clusters with C values are present in the master_file
# CC_clusters_accepted = []
# for i in range(len(master_file)):
#     for j in range(len(thesis_table)):
#         if master_file['Cluster'].str.casefold()[i] == thesis_table['Cluster'].str.casefold()[j]:
#             CC_clusters_accepted.append(thesis_table['Cluster'][j])
# len(CC_clusters_accepted)
# 
# for i in range(len(CC_clusters_accepted)):
#     thesis_table.drop(thesis_table[thesis_table['Cluster']==f'{CC_clusters_accepted[i]}'].index, inplace=True)
# 
# =============================================================================



cluster_total = pd.merge(master_file, thesis_table, left_on = master_file['Cluster'].str.casefold(), right_on = thesis_table['Cluster'].str.casefold(), how = 'inner')
g = cluster_total.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')

# =============================================================================
# mass = cluster_total[cluster_total['M_weighted_1e14'] < 1]
# g = mass.groupby('label')
# CC_clusters = g.get_group('CC')
# NCC_clusters = g.get_group('NCC')
# 
# =============================================================================

Z = (CC_clusters['z'])
E = np.empty(len(CC_clusters['z']))
#USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7
Lx = CC_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_c = CC_clusters['Lx(1e44)']/E
T_c = CC_clusters['T(keV)']
T_new_c = T_c/4.5

log_Lx_c = np.log10(Lx_new_c)
log_T_c = np.log10(T_c)
log_T_new_c = np.log10(T_new_c)
sigma_Lx_c = 0.4343*CC_clusters['eL(%)']/100
sigma_T_c = 0.4343*((CC_clusters['Tmax']-CC_clusters['Tmin'])/(2*T_c))
err_Lx_c = CC_clusters['eL(%)']*Lx_new_c/100
err_T_c = [T_c-(CC_clusters['Tmin']), (CC_clusters['Tmax']-T_c)]
ycept_c, Norm_c, Slope_c, Scatter_c = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)

# Bootstrap for CC #
# =============================================================================
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = CC_clusters.sample(n = len(CC_clusters), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
#     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_CC_BCES.csv')  
# =============================================================================
# =============================================================================



# NCC clusters
Z = (NCC_clusters['z'])
Lx = NCC_clusters['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_n = NCC_clusters['Lx(1e44)']/E
T_n = NCC_clusters['T(keV)']
T_new_n = T_n/4.5

log_Lx_n = np.log10(Lx_new_n)
log_T_n = np.log10(T_n)
log_T_new_n = np.log10(T_new_n)
sigma_Lx_n = 0.4343*NCC_clusters['eL(%)']/100
sigma_T_n = 0.4343*((NCC_clusters['Tmax']-NCC_clusters['Tmin'])/(2*T_n))
err_Lx_n = NCC_clusters['eL(%)']*Lx_new_n/100
err_T_n = [T_n-(NCC_clusters['Tmin']), (NCC_clusters['Tmax']-T_n)]
ycept_n, Norm_n, Slope_n, Scatter_n = general_functions.calculate_bestfit(log_T_new_n, sigma_T_n, log_Lx_n, sigma_Lx_n)

c_c  = CC_clusters['c']
c_n  = NCC_clusters['c']

weights_c = np.ones_like(c_c)/len(c_c)
weights_n = np.ones_like(c_n)/len(c_n)

plt.hist(c_c, bins = 15, label='CC')
plt.hist(c_n, bins = 15, label='NCC')
plt.legend()
plt.show()

len(CC_clusters)/(len(CC_clusters)+len(NCC_clusters))
len(CC_clusters)+len(NCC_clusters)
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = NCC_clusters.sample(n = len(NCC_clusters), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
#     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_NCC_BCES.csv') 
# =============================================================================

# Reading bootstrap data
data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_CC_BCES.csv')
norm_c = data_c['Normalization']
slope_c = data_c['Slope']
scatter_c = data_c['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c = general_functions.calculate_asymm_err(scatter_c)



data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_NCC_BCES.csv')
norm_n = data_n['Normalization']
slope_n = data_n['Slope']
scatter_n = data_n['Scatter']

errnorm_n = general_functions.calculate_asymm_err(norm_n)
errslope_n = general_functions.calculate_asymm_err(slope_n)
errscatter_n = general_functions.calculate_asymm_err(scatter_n)

print('CC best fits:')
print(f'Normalization : {np.round(Norm_c,3)} +/- {np.round(errnorm_c,3)}')
print(f'Slope : {np.round(Slope_c,3)} +/- {np.round(errslope_c,3)}')
print(f'Scatter: {np.round(Scatter_c,3)} +/- {np.round(errscatter_c,3)}')

print('NCC best fits:')

print(f'Normalization : {np.round(Norm_n,3)} +/- {np.round(errnorm_n,3)}')
print(f'Slope : {np.round(Slope_n,3)} +/- {np.round(errslope_n,3)}')
print(f'Scatter: {np.round(Scatter_n,3)} +/- {np.round(errscatter_n,3)}')

sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_c, Slope_c)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_n, Slope_n)

plt.plot(T_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(T_linspace,z_n,label='Best fit NCC', color ='black')


plt.errorbar(T_c, Lx_new_c, xerr=err_T_c, color = 'green', yerr=err_Lx_c, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'CC clusters ({len(T_c)})')
plt.errorbar(T_n, Lx_new_n, xerr=err_T_n, color='red', yerr=err_Lx_n, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'NCC clusters ({len(T_n)})')
#plt.fill_between(T,lcb,ucb, facecolor='red')
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg \,s^{-1}}$)')
plt.title('$L_{\mathrm{X}}-T$ best fit ')
plt.legend(loc = 'lower right')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Lx-T_ccVncc_bestfit.png',dpi=300, bbox_inches="tight")
plt.show()




print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c,errnorm_c,Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c,errslope_c,Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c,errscatter_c,Scatter_n,errscatter_n)}')



print(general_functions.percent_diff(Norm_c,errnorm_c,Norm_n,errnorm_n,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_c,errslope_c,Slope_n,errslope_n,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_c,errscatter_c,Scatter_n,errscatter_n,bestfit_Scatter, err_bestfit_Scatter))


# =============================================================================
# sns.set_context('paper')
# fig, ax_plot = plt.subplots()
# #ax.scatter(slope_c,norm_c)
# general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=1,label=r'CC contour', edgecolor='green', lw = 1)
# general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=3, edgecolor='green', lw = 1)
# plt.scatter(Slope_c,Norm_c,color = 'green', label='CC bestfit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
#      
# general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=1,label=r'NCC contour', edgecolor='darkorange', lw = 1)
# general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
# plt.scatter(Slope_n,Norm_n,color = 'darkorange', label = 'NCC bestfit')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
# 
# 
# plt.xlim(1.50,3)
# plt.ylim(0.5,3.2)
# plt.legend(prop = {'size' : 8})
# plt.xlabel('Slope')
# plt.ylabel('Normalization')
# plt.title('$L_{X}-T$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC ')
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Lx-T_ccVncc_contours_full_sample.png', dpi = 300, bbox_inches="tight")
# plt.show()
# 
# =============================================================================



## Cutting galaxy groups based on Mass  #############

bestfit_Norm_clusters = bestfit_values['Norm_clusters'][0]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][0]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][0]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][0]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][0]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][0]

cluster_total = general_functions.removing_galaxy_groups(cluster_total)
g = cluster_total.groupby('label')
CC_clusters = g.get_group('CC')
NCC_clusters = g.get_group('NCC')



c_new = CC_clusters
Z = (c_new['z'])
#USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7

Lx_c = c_new['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_c = Lx_c/E
T_c = c_new['T(keV)']
T_new_c = T_c/4.5

log_Lx_c = np.log10(Lx_new_c)
log_T_new_c = np.log10(T_new_c)
sigma_Lx_c = 0.4343*c_new['eL(%)']/100
sigma_T_c = 0.4343*((c_new['Tmax']-c_new['Tmin'])/(2*T_c))
err_Lx_c = c_new['eL(%)']*Lx_new_c/100
err_T_c = [T_c-(c_new['Tmin']), (c_new['Tmax']-T_c)]
ycept_c_Mcut, Norm_c_Mcut, Slope_c_Mcut, Scatter_c_Mcut = general_functions.calculate_bestfit(log_T_new_c, sigma_T_c, log_Lx_c, sigma_Lx_c)

# PERFORM BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
#     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_CC(M_cut)_BCES.csv')  
# =============================================================================


# NCC clusters
n_new = NCC_clusters
Z = (n_new['z'])
Lx_n = n_new['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_n = Lx_n/E
T_n = n_new['T(keV)']
T_new_n = T_n/4.5

log_Lx_n = np.log10(Lx_new_n)
log_T_new_n = np.log10(T_new_n)
sigma_Lx_n = 0.4343*n_new['eL(%)']/100
sigma_T_n = 0.4343*((n_new['Tmax']-n_new['Tmin'])/(2*T_n))
err_Lx_n = n_new['eL(%)']*Lx_new_n/100
err_T_n = [T_n-n_new['Tmin'], (n_new['Tmax']-T_n)]
ycept_n_Mcut, Norm_n_Mcut, Slope_n_Mcut, Scatter_n_Mcut = general_functions.calculate_bestfit(log_T_new_n, sigma_T_n, log_Lx_n, sigma_Lx_n)

# PERFORM BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# 
#     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
#     
#     best_A.append(norm)
#     best_B.append(slope)
#     best_scatter.append(scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_NCC(M_cut)_BCES.csv')  
# =============================================================================

data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_CC(M_cut)_BCES.csv')
norm_c_Mcut = data_c['Normalization']
slope_c_Mcut = data_c['Slope']
scatter_c_Mcut = data_c['Scatter']

errnorm_c_Mcut = general_functions.calculate_asymm_err(norm_c_Mcut)
errslope_c_Mcut = general_functions.calculate_asymm_err(slope_c_Mcut)
errscatter_c_Mcut = general_functions.calculate_asymm_err(scatter_c_Mcut)

data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_NCC(M_cut)_BCES.csv')

norm_n_Mcut = data_n['Normalization']
slope_n_Mcut = data_n['Slope']
scatter_n_Mcut = data_n['Scatter']

errnorm_n_Mcut = general_functions.calculate_asymm_err(norm_n_Mcut)
errslope_n_Mcut = general_functions.calculate_asymm_err(slope_n_Mcut)
errscatter_n_Mcut = general_functions.calculate_asymm_err(scatter_n_Mcut)


sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z_c = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_c_Mcut, Slope_c_Mcut)
z_n = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_n_Mcut, Slope_n_Mcut)

plt.plot(T_linspace,z_c,label='Best fit CC', color ='blue')
plt.plot(T_linspace,z_n,label='Best fit NCC', color ='black')


plt.errorbar(T_c, Lx_new_c, xerr=err_T_c, color = 'green', yerr=err_Lx_c, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'CC clusters ({len(T_c)})')
plt.errorbar(T_n, Lx_new_n, xerr=err_T_n, color='red', yerr=err_Lx_n, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'NCC clusters ({len(T_n)})')
#plt.fill_between(T,lcb,ucb, facecolor='red')
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg \,s^{-1}}$)')
plt.title('$L_{\mathrm{X}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$) ')
plt.legend(loc = 'lower right')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Lx-T_ccVncc_bestfit_Mcut.png',dpi=300, bbox_inches="tight")
plt.show()

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut)}')

print(general_functions.percent_diff(Norm_c_Mcut,errnorm_c_Mcut,Norm_n_Mcut,errnorm_n_Mcut,bestfit_Norm_clusters, err_bestfit_Norm_clusters))
print(general_functions.percent_diff(Slope_c_Mcut,errslope_c_Mcut,Slope_n_Mcut,errslope_n_Mcut,bestfit_Slope_clusters, err_bestfit_Slope_clusters))
print(general_functions.percent_diff(Scatter_c_Mcut,errscatter_c_Mcut,Scatter_n_Mcut,errscatter_n_Mcut,bestfit_Scatter_clusters, err_bestfit_Scatter_clusters))

sns.set_context('paper')
fig, ax_plot = plt.subplots()
#ax.scatter(slope_c,norm_c)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=1,label=r'CC (all clusters)', edgecolor='green', lw = 1)
general_functions.confidence_ellipse(slope_c, norm_c, Slope_c, Norm_c, ax_plot, n_std=3, edgecolor='green', lw = 1)
plt.scatter(Slope_c,Norm_c,color = 'green')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
     
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=1,label=r'NCC (all clusters)', edgecolor='darkorange', lw = 1)
general_functions.confidence_ellipse(slope_n, norm_n, Slope_n, Norm_n, ax_plot, n_std=3, edgecolor='darkorange', lw = 1)
plt.scatter(Slope_n,Norm_n,color = 'darkorange')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=1,label=r'CC (clusters+groups)', edgecolor='blue', lw = 1)
general_functions.confidence_ellipse(slope_c_Mcut, norm_c_Mcut, Slope_c_Mcut, Norm_c_Mcut, ax_plot, n_std=3, edgecolor='blue', lw = 1)
plt.scatter(Slope_c_Mcut,Norm_c_Mcut,color = 'blue')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')

general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=1,label=r'NCC (clusters+groups)', edgecolor='red', lw = 1)
general_functions.confidence_ellipse(slope_n_Mcut, norm_n_Mcut, Slope_n_Mcut, Norm_n_Mcut, ax_plot, n_std=3, edgecolor='red', lw = 1)
plt.scatter(Slope_n_Mcut,Norm_n_Mcut,color = 'red')#,label = f'Best fit ({np.round(Slope_,y_bestfit})')
    
plt.xlim(1.0,3)
plt.ylim(0.5,3.5)
plt.legend(prop = {'size' : 8})
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$L_{\mathrm{X}}-T$ : 1$\sigma$ & 3$\sigma$ contours for CC-NCC clusters ')
#plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Lx-T_ccVncc_contours.png', dpi = 300, bbox_inches="tight")
plt.show()





## Cutting galaxy groups based on temperature #######

# =============================================================================
# c_new = CC_clusters[CC_clusters['T(keV)'] > 2]
# Z = (c_new['z'])
# 
# Lx_c = c_new['Lx(1e44)']
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# Lx_new_c = Lx_c/E
# T_c = c_new['T(keV)']
# T_new_c = T_c/4.5
# 
# log_Lx_c = np.log10(Lx_new_c)
# log_T_new_c = np.log10(T_new_c)
# sigma_Lx_c = 0.4343*c_new['eL(%)']/100
# sigma_T_c = 0.4343*((c_new['Tmax']-c_new['Tmin'])/(2*T_c))
# err_Lx_c = c_new['eL(%)']*Lx_new_c/100
# err_T_c = [T_c-(c_new['Tmin']), (c_new['Tmin']-T_c)]
# ycept_c_Tcut, Norm_c_Tcut, Slope_c_Tcut, Scatter_c_Tcut = general_functions.calculate_bestfit(log_T_new_c, sigma_T_c, log_Lx_c, sigma_Lx_c)
# 
# # PERFORM BOOTSTRAP
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # # #cluster_total_CC = cluster_total_CC.to_pandas()
# # for j in range(0,10000):
# #     random_clusters = c_new.sample(n = len(c_new), replace = True)
# #     
# #     Z = (random_clusters['z'])
# #     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
# #     omega_m = 0.3
# #     omega_lambda = 0.7
# # 
# #     Lx = random_clusters['Lx(1e44)']
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     Lx_new = random_clusters['Lx(1e44)']/E
# #     T = random_clusters['T(keV)']
# #     T_new = T/4.5
# # 
# #     log_Lx_c = np.log10(Lx_new)
# #     log_T_c = np.log10(T)
# #     log_T_new_c = np.log10(T_new)
# #     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
# #     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# # 
# #     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
# #     
# #     best_A.append(norm)
# #     best_B.append(slope)
# #     best_scatter.append(scatter)
# # 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# # bestfit_bootstrap.to_csv('Lx-T_CC(2keV)_BCES.csv')  
# # =============================================================================
# 
# 
# # NCC clusters
# n_new = NCC_clusters[NCC_clusters['T(keV)'] > 2]
# Z = (n_new['z'])
# # USING DENSITY PARAMETERS ACCORDING TO LCDM
# omega_m = 0.3
# omega_lambda = 0.7
# 
# Lx_n = n_new['Lx(1e44)']
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# Lx_new_n = Lx_n/E
# T_n = n_new['T(keV)']
# T_new_n = T_n/4.5
# 
# log_Lx_n = np.log10(Lx_new_n)
# log_T_new_n = np.log10(T_new_n)
# sigma_Lx_n = 0.4343*n_new['eL(%)']/100
# sigma_T_n = 0.4343*((n_new['Tmax']-n_new['Tmin'])/(2*T_n))
# err_Lx_n = n_new['eL(%)']*Lx_new_n/100
# err_T_n = [T_n-n_new['Tmin'], (n_new['Tmax']-T_n)]
# ycept_n_Tcut, Norm_n_Tcut, Slope_n_Tcut, Scatter_n_Tcut = general_functions.calculate_bestfit(log_T_new_n, sigma_T_n, log_Lx_n, sigma_Lx_n)
# 
# # PERFORM BOOTSTRAP
# # =============================================================================
# # best_A = []
# # best_B = []
# # best_scatter = []
# # # #cluster_total_CC = cluster_total_CC.to_pandas()
# # for j in range(0,10000):
# #     random_clusters = n_new.sample(n = len(n_new), replace = True)
# #     
# #     Z = (random_clusters['z'])
# #     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
# #     omega_m = 0.3
# #     omega_lambda = 0.7
# # 
# #     Lx = random_clusters['Lx(1e44)']
# #     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# #     Lx_new = random_clusters['Lx(1e44)']/E
# #     T = random_clusters['T(keV)']
# #     T_new = T/4.5
# # 
# #     log_Lx_c = np.log10(Lx_new)
# #     log_T_c = np.log10(T)
# #     log_T_new_c = np.log10(T_new)
# #     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
# #     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
# # 
# #     yintercept,norm,slope,scatter = general_functions.calculate_bestfit(log_T_new_c,sigma_T_c,log_Lx_c,sigma_Lx_c)
# #     
# #     best_A.append(norm)
# #     best_B.append(slope)
# #     best_scatter.append(scatter)
# # 
# # bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# # bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# # bestfit_bootstrap.to_csv('Lx-T_NCC(2keV)_BCES.csv')  
# # =============================================================================
# 
# data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_CC(2keV)_BCES.csv')
# norm_c_Tcut = data_c['Normalization']
# slope_c_Tcut = data_c['Slope']
# scatter_c_Tcut = data_c['Scatter']
# 
# errnorm_c_Tcut = general_functions.calculate_asymm_err(norm_c_Tcut)
# errslope_c_Tcut = general_functions.calculate_asymm_err(slope_c_Tcut)
# errscatter_c_Tcut = general_functions.calculate_asymm_err(scatter_c_Tcut)
# 
# data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_NCC(2keV)_BCES.csv')
# 
# norm_n_Tcut = data_n['Normalization']
# slope_n_Tcut = data_n['Slope']
# scatter_n_Tcut = data_n['Scatter']
# 
# errnorm_n_Tcut = general_functions.calculate_asymm_err(norm_n_Tcut)
# errslope_n_Tcut = general_functions.calculate_asymm_err(slope_n_Tcut)
# errscatter_n_Tcut = general_functions.calculate_asymm_err(scatter_n_Tcut)
# 
# 
# sns.set_context('paper')
# plt.errorbar(T_c, Lx_new_c, xerr=err_T_c, color = 'green', yerr=err_Lx_c, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'CC clusters ({len(T_c)})')
# plt.errorbar(T_n, Lx_new_n, xerr=err_T_n, color='red', yerr=err_Lx_n, ls='', fmt='.', capsize=2, alpha=1, elinewidth=0.6, label=f'NCC clusters ({len(T_n)})')
# 
# z_c = Norm_c_Tcut * T_new_c ** Slope_c_Tcut
# z_n = Norm_n_Tcut * T_new_n ** Slope_n_Tcut
# 
# plt.plot(T_c, z_c, color='blue', label='CC bestfit')
# plt.plot(T_n, z_n, color='black', label='NCC bestfit')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.7,25)
# plt.ylim(0.005, 45)
# #plt.axvline(2, color='black', ls='--')
# 
# plt.legend(loc = 'lower right')
# plt.axvline(2, color = 'black', ls = '--')
# plt.xlabel('T [keV]')
# plt.ylabel('$L_{X}$/E(z) *$10^{44}$ [erg/s]')
# plt.title('$L_{X}-T$ best fit (T>2keV )')
# plt.legend(loc = 'best')
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/CC-NCC_comparison/Lx-T_ccVncc_Tcut_bestfit.png',dpi=300, bbox_inches="tight")
# plt.show()
# 
# print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_c_Tcut,errnorm_c_Tcut,Norm_n_Tcut,errnorm_n_Tcut)}')
# print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_c_Tcut,errslope_c_Tcut,Slope_n_Tcut,errslope_n_Tcut)}')
# print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_c_Tcut,errscatter_c_Tcut,Scatter_n_Tcut,errscatter_n_Tcut)}')
# 
# bestfit_Norm_Tcut = 1.438
# err_bestfit_Norm_Tcut = 0.001
# bestfit_Slope_Tcut = 2.045
# err_bestfit_Slope_Tcut = 0.005
# bestfit_Scatter_Tcut = 0.222
# err_bestfit_Scatter_Tcut = 0.001
# 
# print(general_functions.percent_diff(Norm_c_Tcut,errnorm_c_Tcut,Norm_n_Tcut,errnorm_n_Tcut,bestfit_Norm_Tcut, err_bestfit_Norm_Tcut))
# print(general_functions.percent_diff(Slope_c_Tcut,errslope_c_Tcut,Slope_n_Tcut,errslope_n_Tcut,bestfit_Slope_Tcut, err_bestfit_Slope_Tcut))
# print(general_functions.percent_diff(Scatter_c_Tcut,errscatter_c_Tcut,Scatter_n_Tcut,errscatter_n_Tcut,bestfit_Scatter_Tcut, err_bestfit_Scatter_Tcut))
# 
# 
# T_c = CC_clusters['T(keV)']
# T_n = NCC_clusters['T(keV)']
# bins_c, cdf_c = general_functions.calculate_cdf(T_c, 20)
# bins_n, cdf_n = general_functions.calculate_cdf(T_n, 20)
# plt.plot(bins_c[1:], cdf_c,label = f'CC ({len(T_c)})')
# plt.plot(bins_n[1:], cdf_n, label = f'NCC ({len(T_n)})')
# plt.xlabel('T [keV]')
# plt.ylabel('CDF')
# plt.title('CDF for T')
# plt.legend(loc='best')
# #plt.savefig('CDF_T_ccVncc.png',dpi = 300)
# plt.show()
# general_functions.calculate_ks_stat(T_c, T_n)
# 
# 
# C_c = c_new['c']
# err_c_c = c_new['e_c']
# z_c = ycept_c + Slope_c* log_T_new_c
# z_n = ycept_n + Slope_n* log_T_new_n
# 
# C_n = n_new['c']
# err_c_n = n_new['e_c']
# plt.errorbar(C_c,z_c-log_Lx_c, yerr = sigma_Lx_c, xerr= err_c_c ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='green', label=f'CC Clusters ({len(z_c)})')
# plt.errorbar(C_n,z_n-log_Lx_n, yerr = sigma_Lx_n, xerr= err_c_n ,ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'NCC Clusters ({len(z_n)})')
# plt.ylim(-1,1)
# plt.xlim(-0.1, 0.8)
# plt.xlabel('Concentration')
# plt.ylabel('$\Delta log_{10}L_{X}$')
# plt.title('$L_{X}-T$ residuals')
# plt.legend(loc = 'best')
# plt.axhline(0, color = 'black')
# plt.axvline(0.18, color= 'blue', ls= '--')
# # plt.savefig('R-T_best_fit-NCC.png',dpi=300)
# plt.show()
# 
# CC_clusters.iloc[0]
# =============================================================================

#########   SCaling relation with C ###########################################
               #For CC clusters     
#############################################################################################
c_new = CC_clusters
Z = (c_new['z'])
#USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7
CC_clusters.iloc[0]
Lx_c = c_new['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_c = Lx_c/E
T_c = c_new['T(keV)']
T_new_c = T_c/4.5

log_Lx_c = np.log10(Lx_new_c)
log_T_new_c = np.log10(T_new_c)
sigma_Lx_c = 0.4343*c_new['eL(%)']/100
sigma_T_c = 0.4343*((c_new['Tmax']-c_new['Tmin'])/(2*T_c))
err_Lx_c = c_new['eL(%)']*Lx_new_c/100
err_T_c = [T_c-(c_new['Tmin']), (c_new['Tmin']-T_c)]

c_c = c_new['c']/np.median(c_new['c'])
e_c_c = c_new['e_c']
log_c_c = np.log10(c_c)
sigma_c_c = 0.4343 * e_c_c/c_c

cov = np.cov(sigma_Lx_c,sigma_c_c)

yarray_c = log_Lx_c - 0.32*log_c_c
xarray_c = log_T_new_c
yerr_c = np.sqrt( (sigma_Lx_c)**2 + (0.32*sigma_c_c)**2 ) - 2*0.32*cov[0][1]
xerr_c = sigma_T_c 
test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray_c,xerr_c,yarray_c, yerr_c)

#PERFORM BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = c_new.sample(n = len(c_new), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_c = np.log10(Lx_new)
#     log_T_c = np.log10(T)
#     log_T_new_c = np.log10(T_new)
#     sigma_Lx_c = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_c = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     
#     c_c = random_clusters['c']/np.median(random_clusters['c'])
#     e_c_c = random_clusters['e_c']
#     log_c_c = np.log10(c_c)
#     sigma_c_c = 0.4343 * e_c_c/c_c
#     
#     cov = np.cov(sigma_Lx_c,sigma_c_c)
# 
#     yarray_c = log_Lx_c - 0.32*log_c_c
#     xarray_c = log_T_new_c
#     yerr_c = np.sqrt( (sigma_Lx_c)**2 + (0.32*sigma_c_c)**2 ) - 2*0.32*cov[0][1]
#     xerr_c = sigma_T_c 
#     test_Ycept_c, test_Norm_c, test_Slope_c, test_Scatter_c = general_functions.calculate_bestfit(xarray_c,xerr_c,yarray_c, yerr_c)
# 
#     best_A.append(test_Norm_c)
#     best_B.append(test_Slope_c)
#     best_scatter.append(test_Scatter_c)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T_CC(c_inc)_BCES.csv')  
# =============================================================================

#########################################################################################
                            #For NCC clusters
n_new = NCC_clusters
Z = (n_new['z'])
# USING DENSITY PARAMETERS ACCORDING TO LCDM
omega_m = 0.3
omega_lambda = 0.7

Lx_n = n_new['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new_n = Lx_n/E
T_n = n_new['T(keV)']
T_new_n = T_n/4.5

log_Lx_n = np.log10(Lx_new_n)
log_T_new_n = np.log10(T_new_n)
sigma_Lx_n = 0.4343*n_new['eL(%)']/100
sigma_T_n = 0.4343*((n_new['Tmax']-n_new['Tmin'])/(2*T_n))
err_Lx_n = n_new['eL(%)']*Lx_new_n/100
err_T_n = [T_n-n_new['Tmin'], (n_new['Tmax']-T_n)]

c_n = n_new['c']/np.median(n_new['c'])
e_c_n = n_new['e_c']
log_c_n = np.log10(c_n)
sigma_c_n = 0.4343 * e_c_n/c_n

cov = np.cov(sigma_Lx_n,sigma_c_n)
yarray_n = log_Lx_n - 0.32*log_c_n
xarray_n = log_T_new_n
yerr_n = np.sqrt( (sigma_Lx_n)**2 + (0.32*sigma_c_n)**2 ) -  2*0.32*cov[0][1]
xerr_n = sigma_T_n 
test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray_n,xerr_n,yarray_n, yerr_n)


#PERFORM BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     random_clusters = n_new.sample(n = len(n_new), replace = True)
#     
#     Z = (random_clusters['z'])
#     ##USING DENSITY PARAMETERS ACCORDING TO LCDM
#     omega_m = 0.3
#     omega_lambda = 0.7
# 
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = random_clusters['Lx(1e44)']/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
# 
#     log_Lx_n = np.log10(Lx_new)
#     log_T_n = np.log10(T)
#     log_T_new_n = np.log10(T_new)
#     sigma_Lx_n = 0.4343*random_clusters['eL(%)']/100
#     sigma_T_n = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     
#     c_n = random_clusters['c']/np.median(random_clusters['c'])
#     e_c_n = random_clusters['e_c']
#     log_c_n = np.log10(c_n)
#     sigma_c_n = 0.4343 * e_c_n/c_n
#     
#     cov = np.cov(sigma_Lx_n,sigma_c_n)
# 
#     yarray_n = log_Lx_n - 0.32*log_c_n
#     xarray_n = log_T_new_n
#     yerr_n = np.sqrt( (sigma_Lx_n)**2 + (0.32*sigma_c_n)**2 ) - 2*0.32*cov[0][1]
#     xerr_n = sigma_T_n 
#     test_Ycept_n, test_Norm_n, test_Slope_n, test_Scatter_n = general_functions.calculate_bestfit(xarray_n,xerr_n,yarray_n, yerr_n)
# 
#     best_A.append(test_Norm_n)
#     best_B.append(test_Slope_n)
#     best_scatter.append(test_Scatter_n)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-T(c_inc)_NCC_BCES.csv')  
# =============================================================================


data_c = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T(c_inc)_CC_BCES.csv')
norm_c = data_c['Normalization']
slope_c = data_c['Slope']
scatter_c = data_c['Scatter']

errnorm_c = general_functions.calculate_asymm_err(norm_c)
errslope_c = general_functions.calculate_asymm_err(slope_c)
errscatter_c = general_functions.calculate_asymm_err(scatter_c)

data_n = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T(c_inc)_NCC_BCES.csv')
norm_n = data_n['Normalization']
slope_n = data_n['Slope']
scatter_n = data_n['Scatter']

errnorm_n = general_functions.calculate_asymm_err(norm_n)
errslope_n = general_functions.calculate_asymm_err(slope_n)
errscatter_n = general_functions.calculate_asymm_err(scatter_n)



# =============================================================================
# Normalization : 1.438 +/- [-0.064  0.073]
# Slope : 2.045 +/- [-0.067  0.067]
# Scatter: 0.222 +/- [-0.014  0.014]
# =============================================================================

print(f'Normalization sigma : {general_functions.calculate_sigma_dev(test_Norm_n,errnorm_c,test_Norm_n,errnorm_n)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(test_Slope_n,errslope_c,test_Slope_n,errslope_n)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(test_Scatter_n,errscatter_c,test_Scatter_n,errscatter_n)}')



