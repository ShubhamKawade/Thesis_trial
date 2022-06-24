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
bestfit_Norm = bestfit_values['Norm_all'][3]
err_bestfit_Norm = bestfit_values['err_Norm_all'][3]
bestfit_Slope = bestfit_values['Slope_all'][3]
err_bestfit_Slope = bestfit_values['err_Slope_all'][3]
bestfit_Scatter = bestfit_values['Scatter_all'][3]
err_bestfit_Scatter = bestfit_values['err_Scatter_all'][3]

bcgt = pd.read_csv('/home/schubham/Thesis/Thesis/Data/Lx-BCG-Ysz-full-eeHIFL_mass.csv')
bcgt = general_functions.cleanup(bcgt)
bcgt = bcgt[(bcgt['z']>0.03) & (bcgt['z']< 0.15) ]
offset = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_FINAL_ANISOTROPY_BCG_OFFSET.csv')
offset = general_functions.cleanup(offset)

L_sun = 1
Lbcg = L_sun * 10 ** (0.4*(3.27 - bcgt['BCGMag']))
bcgt['Lbcg'] = Lbcg

bcgt_all = pd.merge(bcgt,offset, left_on = bcgt['Cluster'].str.casefold(), right_on = offset['Cluster'].str.casefold(), how ='inner')
r_clusters = bcgt_all[bcgt_all['BCG_offset_R500'] < 0.01]
d_clusters = bcgt_all[bcgt_all['BCG_offset_R500'] > 0.08 ]

# Relaxed clusters
Lx_r = r_clusters['Lx']
log_Lx_r = np.log10(Lx_r)
sigma_Lx_r = 0.4343*r_clusters['eL(%)']/100
err_Lx_r = r_clusters['eL(%)']*Lx_r/100

Lbcg_r = r_clusters['Lbcg']
log_Lbcg_r = np.log10(Lbcg_r)
Lbcg_new_r = Lbcg_r / 6e11
log_Lbcg_new_r = np.log10(Lbcg_new_r)
sigma_Lbcg_r = np.zeros(len(sigma_Lx_r))
ycept_r,Norm_r,Slope_r,Scatter_r = general_functions.calculate_bestfit(log_Lbcg_new_r,sigma_Lbcg_r,log_Lx_r,sigma_Lx_r)

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters =r_clusters.sample(n = len(r_clusters), replace = True)
#     
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_r_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_r_BCES.csv')
norm_r = data['Normalization']
slope_r = data['Slope']
scatter_r = data['Scatter']

errnorm_r =  general_functions.calculate_asymm_err(norm_r)
errslope_r = general_functions.calculate_asymm_err(slope_r)
errscatter_r =  general_functions.calculate_asymm_err(scatter_r)


# Disturbed clusters
Lx_d = d_clusters['Lx']
log_Lx_d = np.log10(Lx_d)
sigma_Lx_d = 0.4343*d_clusters['eL(%)']/100
err_Lx_d = d_clusters['eL(%)']*Lx_d/100

Lbcg_d = d_clusters['Lbcg']
log_Lbcg_d = np.log10(Lbcg_d)
Lbcg_new_d = Lbcg_d / 6e11
log_Lbcg_new_d = np.log10(Lbcg_new_d)
sigma_Lbcg_d = np.zeros(len(sigma_Lx_d))
ycept_d,Norm_d,Slope_d,Scatter_d = general_functions.calculate_bestfit(log_Lbcg_new_d,sigma_Lbcg_d,log_Lx_d,sigma_Lx_d)

#Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
#   
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
# 
# 
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_d_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_d_BCES.csv')
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



sns.set_context('paper')
Lbcg_linspace = np.linspace(1,25,100)

z_r = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept_r, Slope_r)
z_d = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept_d, Slope_d)

plt.plot(Lbcg_linspace,z_r, label = 'Best fit relaxed',color = 'blue')
plt.plot(Lbcg_linspace,z_d, label = 'Best fit disturbed',color = 'black')
plt.errorbar(Lbcg_r/1e11,Lx_r,yerr = err_Lx_r,color = 'green',ls='',fmt='.', capsize = 3 ,alpha= 0.7, elinewidth = 0.5, label = f'relaxed clusters ({len(Lx_r)})' )
plt.errorbar(Lbcg_d/1e11,Lx_d,yerr = err_Lx_d,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'disturbed clusters ({len(Lx_d)})' )
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$L_{\mathrm{X}}$ (*$10^{44}$ erg/s) ')
plt.xlabel('$L_{\mathrm{BCG}}$  (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.title('$L_{\mathrm{X}}-L_{\mathrm{BCG}}$ best fit')
plt.xlim(1,25)
plt.ylim(0.004,60)
plt.legend(loc = 'lower right')

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/Lx-Lbcg_rVd_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()


print(f'Normalization sigma : {general_functions.calculate_sigma_dev(Norm_r,errnorm_r,Norm_d,errnorm_d)}')
print(f'Slope sigma : {general_functions.calculate_sigma_dev(Slope_r,errslope_r,Slope_d,errslope_d)}')
print(f'Scatter sigma : {general_functions.calculate_sigma_dev(Scatter_r,errscatter_r,Scatter_d,errscatter_d)}')


print(general_functions.percent_diff(Norm_r,errnorm_r,Norm_d,errnorm_d,bestfit_Norm, err_bestfit_Norm))
print(general_functions.percent_diff(Slope_r,errslope_r,Slope_d,errslope_d,bestfit_Slope, err_bestfit_Slope))
print(general_functions.percent_diff(Scatter_r,errscatter_r,Scatter_d,errscatter_d,bestfit_Scatter, err_bestfit_Scatter))



##########################################################


   # Cutting galaxy groups based on mass


##########################################################

bestfit_Norm_clusters = bestfit_values['Norm_clusters'][3]
err_bestfit_Norm_clusters = bestfit_values['err_Norm_clusters'][3]
bestfit_Slope_clusters = bestfit_values['Slope_clusters'][3]
err_bestfit_Slope_clusters = bestfit_values['err_Slope_clusters'][3]
bestfit_Scatter_clusters = bestfit_values['Scatter_clusters'][3]
err_bestfit_Scatter_clusters = bestfit_values['err_Scatter_clusters'][3]

# Relaxed clusters
r_clusters = general_functions.removing_galaxy_groups(r_clusters)
Lx_r = r_clusters['Lx']
log_Lx_r = np.log10(Lx_r)
sigma_Lx_r = 0.4343*r_clusters['eL(%)']/100
err_Lx_r = r_clusters['eL(%)']*Lx_r/100

Lbcg_r = r_clusters['Lbcg']
log_Lbcg_r = np.log10(Lbcg_r)
Lbcg_new_r = Lbcg_r / 6e11
log_Lbcg_new_r = np.log10(Lbcg_new_r)
sigma_Lbcg_r = np.zeros(len(sigma_Lx_r))
ycept_r_Mcut,Norm_r_Mcut,Slope_r_Mcut,Scatter_r_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_r,sigma_Lbcg_r,log_Lx_r,sigma_Lx_r)

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters =r_clusters.sample(n = len(r_clusters), replace = True)
#     
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
# 
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
#     
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_r(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_r(Mcut)_BCES.csv')
norm_r_Mcut = data['Normalization']
slope_r_Mcut = data['Slope']
scatter_r_Mcut = data['Scatter']

errnorm_r_Mcut =  general_functions.calculate_asymm_err(norm_r_Mcut)
errslope_r_Mcut = general_functions.calculate_asymm_err(slope_r_Mcut)
errscatter_r_Mcut =  general_functions.calculate_asymm_err(scatter_r_Mcut)


# Disturbed clusters
d_clusters = general_functions.removing_galaxy_groups(d_clusters)
Lx_d = d_clusters['Lx']
log_Lx_d = np.log10(Lx_d)
sigma_Lx_d = 0.4343*d_clusters['eL(%)']/100
err_Lx_d = d_clusters['eL(%)']*Lx_d/100

Lbcg_d = d_clusters['Lbcg']
log_Lbcg_d = np.log10(Lbcg_d)
Lbcg_new_d = Lbcg_d / 6e11
log_Lbcg_new_d = np.log10(Lbcg_new_d)
sigma_Lbcg_d = np.zeros(len(sigma_Lx_d))
ycept_d_Mcut,Norm_d_Mcut,Slope_d_Mcut,Scatter_d_Mcut = general_functions.calculate_bestfit(log_Lbcg_new_d,sigma_Lbcg_d,log_Lx_d,sigma_Lx_d)

#Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# #cluster_total = cluster_total.to_pandas()
# for j in range(0,10000):
#     random_clusters = d_clusters.sample(n = len(d_clusters), replace = True)
#   
#     Lx = random_clusters['Lx']
#     log_Lx = np.log10(Lx)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
# 
# 
#     Lbcg = random_clusters['Lbcg']
#     log_Lbcg = np.log10(Lbcg)
#     Lbcg_new = Lbcg / 6e11
#     log_Lbcg_new = np.log10(Lbcg_new)
#     sigma_Lbcg = np.zeros(len(sigma_Lx))
#     
#     ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_Lbcg_new,sigma_Lbcg,log_Lx,sigma_Lx)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('Lx-Lbcg_d(Mcut)_BCES.csv')
# =============================================================================

data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-Lbcg_d(Mcut)_BCES.csv')
norm_d_Mcut = data['Normalization']
slope_d_Mcut = data['Slope']
scatter_d_Mcut = data['Scatter']
errnorm_d_Mcut =  general_functions.calculate_asymm_err(norm_d_Mcut)
errslope_d_Mcut = general_functions.calculate_asymm_err(slope_d_Mcut)
errscatter_d_Mcut = general_functions.calculate_asymm_err(scatter_d_Mcut)

sns.set_context('paper')
Lbcg_linspace = np.linspace(1,25,100)

z_r = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept_r_Mcut, Slope_r_Mcut)
z_d = general_functions.plot_bestfit(Lbcg_linspace, 6, 1, ycept_d_Mcut, Slope_d_Mcut)

plt.plot(Lbcg_linspace,z_r, label = 'Best fit relaxed',color = 'blue')
plt.plot(Lbcg_linspace,z_d, label = 'Best fit disturbed',color = 'black')
plt.errorbar(Lbcg_r/1e11,Lx_r,yerr = err_Lx_r,color = 'green',ls='',fmt='.', capsize = 3 ,alpha= 0.7, elinewidth = 0.5, label = f'relaxed clusters ({len(Lx_r)})' )
plt.errorbar(Lbcg_d/1e11,Lx_d,yerr = err_Lx_d,color = 'red',ls='',fmt='.', capsize = 2,alpha= 0.7, elinewidth = 0.6, label = f'disturbed clusters ({len(Lx_d)})' )
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,25)
plt.ylim(0.004,60)
plt.legend(loc = 'lower right')

plt.ylabel('$L_{\mathrm{X}}$ (*$10^{44}$ erg/s) ')
plt.xlabel('$L_{\mathrm{BCG}}$  (*$10^{11}\,\mathrm{L}_{\odot}$)')
plt.title('$L_{\mathrm{X}}-L_{\mathrm{BCG}}$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Relaxed-Disturbed_comparison/Lx-Lbcg_rVd(Mcut)_bestfit.png',dpi=300,bbox_inches="tight")
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
    
plt.xlim(-1,2.5)
plt.ylim(-1,4.0)
plt.legend(prop = {'size' : 8}, loc='lower right')
plt.xlabel('Slope')
plt.ylabel('Normalization')
plt.title('$L_{\mathrm{X}}-L_{\mathrm{BCG}}$ : 1$\sigma$ & 3$\sigma$ contours for relaxed-disturbed')
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Contour_plots/Lx-Lbcg_relaxed_contours.png' ,dpi=300, bbox_inches="tight")

plt.show()





