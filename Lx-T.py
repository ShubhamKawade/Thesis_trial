#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
import seaborn as sns
import bces

cluster_total = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')

omega_m = 0.3
omega_lambda = 0.7
Z = (cluster_total['z'])
np.median(Z)
Lx = cluster_total['Lx(1e44)']
np.max(Lx)
np.median(Lx)
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new = cluster_total['Lx(1e44)']/E
T = cluster_total['T(keV)']
T_new = T/4.5


# =============================================================================
# sns.set_context('paper')
# weights = np.ones_like(Lx)/len(Lx)
# plt.hist(Lx, bins=10, label = 'Normalizations a' )
# #plt.axvline(np.median(Lx), label='median', color = 'black')
# 
# plt.xlabel(r'$L_{\mathrm{X}}$ ($10^{44}$ erg/s)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,300)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Lx_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# 
# 
# sns.set_context('paper')
# weights = np.ones_like(Z)/len(Z)
# plt.hist(Z, bins=10, label = 'NormaliZations a' )
# #plt.axvline(np.median(Z), label='median', color = 'black')
# 
# plt.xlabel('$z$')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normaliZations')
# #plt.legend(loc = 'upper right')
# plt.xlim(-0.03,0.55)
# plt.ylim(0,150)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Z_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# 
# 
# sns.set_context('paper')
# weights = np.ones_like(T)/len(T)
# plt.hist(T, bins=10, label = 'Normalizations a' )
# #plt.axvline(np.median(T), label='median', color = 'black')
# 
# plt.xlabel('T (keV)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,120)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/T_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# =============================================================================


log_Lx = np.log10(Lx_new)
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_Lx = 0.4343*cluster_total['eL(%)']/100
sigma_T = 0.4343*((cluster_total['Tmax']-cluster_total['Tmin'])/(2*T))

err_Lx = cluster_total['eL(%)']*Lx_new/100
err_T = [(T-cluster_total['Tmin']), (cluster_total['Tmax']-T)]

class Scaling_relation:
    
    import numpy as np
    import bces.bces
    from scipy import stats 
    import general_functions as gf

    
    def __init__(self, x_data, norm_x, x_err, y_data, norm_y, y_err ):
        self.x = x_data
        self.y = y_data
        self.x_err = x_err
        self.y_err = y_err
        self.norm_x = norm_x
        self.norm_y = norm_y
        
    
    
    def bestfit(self):
         a, b, a_err, b_err, cov_ab = bces.bces.bces(self.x, self.x_err, self.y, self.y_err,0)
         #scatter = gf.calculate_sig_intr_yx(self.x, self.y, self.x_err, self.y_err, a[0], b[0])

         return b[0], 10 ** b[0], a[0]

Lx_T = Scaling_relation(log_T,4.5,sigma_T,log_Lx,1,sigma_Lx)
Lx_T.bestfit()



ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)

#some example data

# PERFORMING BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0, 10000):
#     random_clusters = cluster_total.sample(n=len(cluster_total), replace=True)
# 
#     Z = (random_clusters['z'])
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = Lx/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     logLx = np.log10(Lx_new)
#     logT = np.log10(T_new)
#     logT_new = np.log10(T_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# bestfit_bootstrap.to_csv('Lx-T_all_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_all_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm = general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter = general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')

# =============================================================================
# sns.set_context('paper')
# weights = np.ones_like(norm)/len(norm)
# plt.hist(norm, bins=10, weights=weights, label = 'Normalizations a' )
# plt.axvline(Norm, label='best fit ', color = 'black')
# plt.axvline(Norm+errnorm[0], label = 'a-1$\sigma$', ls='--',color = 'black')
# plt.axvline(Norm+errnorm[1], label = 'a+1$\sigma$',ls = '--', color = 'black')
# plt.xlabel('Normalizations')
# plt.ylabel('counts')
# plt.title(r'$L_{\mathrm{X}}-T$ bootstrap normalizations')
# plt.legend(loc = 'upper right')
# plt.xlim(1.4201,1.4225)
# plt.ylim(0,0.4)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-T_bootstrap_norm.png',dpi=300,bbox_inches="tight")
# plt.show()
# 
# =============================================================================

sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept, Slope)
plt.plot(T_linspace,z,label='Best fit', color ='green')

plt.errorbar(T, Lx_new, xerr=err_T, yerr=err_Lx,color = 'red', ls='', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'Clusters ({len(T_new)})')
#plt.fill_between(T,lcb,ucb, facecolor='red')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg \,s^{-1}}$)')
plt.title('$L_{\mathrm{X}}-T$ best fit ')
plt.legend(loc = 'lower right')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-T_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

#To calculate BIC
# =============================================================================
# z_log = ycept + log_T_new * Slope
# BIC = general_functions.calculate_BIC(Lx_new, z, 2)
# 
# =============================================================================

############## CUTTING THE GALAXY GROUPS #######################
cluster_total = general_functions.removing_galaxy_groups(cluster_total)

omega_m = 0.3
omega_lambda = 0.7
Z = (cluster_total['z'])
Lx = cluster_total['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new = cluster_total['Lx(1e44)']/E
T = cluster_total['T(keV)']
T_new = T/4.5

log_Lx = np.log10(Lx_new)
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_Lx = 0.4343*cluster_total['eL(%)']/100
sigma_T = 0.4343*((cluster_total['Tmax']-cluster_total['Tmin'])/(2*T))

err_Lx = cluster_total['eL(%)']*Lx_new/100
err_T = [(T-cluster_total['Tmin']), (cluster_total['Tmax']-T)]

ycept_Mcut, Norm_Mcut, Slope_Mcut, Scatter_Mcut = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
sns.set_context('paper')

T_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 1, ycept_Mcut, Slope_Mcut)
plt.plot(T_linspace,z,label='Best fit', color ='green')

plt.errorbar(T, Lx_new, xerr=err_T, yerr=err_Lx,color = 'red', ls='', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'Clusters ({len(T_new)})')
#plt.fill_between(T,lcb,ucb, facecolor='red')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.xlabel('$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}\,E(z)^{-1}$ (*$10^{44} \mathrm{\,erg \,s^{-1}}$)')
plt.title('$L_{\mathrm{X}}-T$ best fit ')
plt.legend(loc = 'lower right')
plt.title(r'$L_{\mathrm{X}}-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}M_{\odot}$)')

plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-T_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()
# PERFORMING BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0, 10000):
#     random_clusters = cluster_total.sample(n=len(cluster_total), replace=True)
# 
#     Z = (random_clusters['z'])
#     E = np.empty(len(random_clusters['z']))
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = Lx/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     logLx = np.log10(Lx_new)
#     logT = np.log10(T_new)
#     logT_new = np.log10(T_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# bestfit_bootstrap.to_csv('Lx-T_all(Mcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_all(Mcut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut = general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut = general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')

############################ making T cut ######################################
cluster_total = cluster_total[cluster_total['T(keV)'] > 2]

omega_m = 0.3
omega_lambda = 0.7
Z = (cluster_total['z'])
Lx = cluster_total['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new = cluster_total['Lx(1e44)']/E
T = cluster_total['T(keV)']
T_new = T/4.5

log_Lx = np.log10(Lx_new)
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_Lx = 0.4343*cluster_total['eL(%)']/100
sigma_T = 0.4343*((cluster_total['Tmax']-cluster_total['Tmin'])/(2*T))

err_Lx = cluster_total['eL(%)']*Lx_new/100
err_T = [(T-cluster_total['Tmin']), (cluster_total['Tmax']-T)]

ycept_Tcut, Norm_Tcut, Slope_Tcut, Scatter_Tcut = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
sns.set_context('paper')
plt.errorbar(T, Lx_new, xerr=err_T, yerr=err_Lx, ls='',color = 'red', fmt='.', capsize=1.7, alpha=0.8, elinewidth=0.65, label=f'Clusters ({len(T_new)})')
z = Norm_Tcut * T_new ** Slope_Tcut
plt.plot(T,z, label='Best fit ', color = 'green')
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$T$ (keV)')
plt.ylabel(r'$L_{\mathrm{X}}$/E(z) (*$10^{44}$ erg/s)')
plt.title(r'$L_{\mathrm{X}}-T$ best fit ($T > 2$ keV)')
plt.legend(loc = 'lower right')
plt.xlim(0.6,25)
plt.ylim(0.003,90)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/Lx-T_Tcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()
# PERFORMING BOOTSTRAP
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0, 10000):
#     random_clusters = cluster_total.sample(n=len(cluster_total), replace=True)
# 
#     Z = (random_clusters['z'])
#     E = np.empty(len(random_clusters['z']))
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = Lx/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     logLx = np.log10(Lx_new)
#     logT = np.log10(T_new)
#     logT_new = np.log10(T_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)
#     best_A.append(Norm)
#     best_B.append(Slope)
#     best_scatter.append(Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# bestfit_bootstrap.to_csv('Lx-T_all(Tcut)_BCES.csv')
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_all(Tcut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Tcut = general_functions.calculate_asymm_err(norm)
errslope_Tcut = general_functions.calculate_asymm_err(slope)
errscatter_Tcut = general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm_Tcut,3)} +/- {np.round(errnorm_Tcut,3)}')
print(f'Slope : {np.round(Slope_Tcut,3)} +/- {np.round(errslope_Tcut,3)}')
print(f'Scatter: {np.round(Scatter_Tcut,3)} +/- {np.round(errscatter_Tcut,3)}')

# Norm = 1.438 [-0.001  0.001]
# slope  2.045 [-0.005  0.006]
# scatter 0.222 [-0.001  0.001]

############## Scaling relation including the concentration parameter #####


thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv',sep = ',')
thesis_table = general_functions.cleanup(thesis_table)
thesis_table.iloc[0]
cluster_total = pd.merge(cluster_total, thesis_table, how = 'inner', left_on='Cluster', right_on = 'Cluster' )

cluster_total = cluster_total.dropna(axis=0,subset = ['c'], inplace = False)
cluster_total.iloc[0]
omega_m = 0.3
omega_lambda = 0.7
Z = (cluster_total['z'])
Lx = cluster_total['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new = cluster_total['Lx(1e44)']/E
T = cluster_total['T(keV)']
T_new = T/4.5

log_Lx = np.log10(Lx_new)
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_Lx = 0.4343*cluster_total['eL(%)']/100
sigma_T = 0.4343*((cluster_total['Tmax']-cluster_total['Tmin'])/(2*T))

err_Lx = cluster_total['eL(%)']*Lx_new/100
err_T = [(T-cluster_total['Tmin']), (cluster_total['Tmax']-T)]
ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)

######## Defining new variables #############
c = cluster_total['c']/np.median(cluster_total['c'])
e_c = cluster_total['e_c']
log_c = np.log10(c)
sigma_c = 0.4343 * e_c/c
### To constrain the  g factor for concentration ##############
# =============================================================================
# g = np.arange(-3,3,0.01)
# test_scatter = []
# test_norm = []
# test_slope = []
# gamma = []
# cov = np.cov(sigma_Lx,sigma_c)
# for i in g:
#     yarray = log_Lx - i*log_c
#     yerr = np.sqrt( (sigma_Lx)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
#     xarray = log_T_new
#     xerr = sigma_T
#     
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#     test_scatter.append(Scatter)
#     test_norm.append(Norm)
#     test_slope.append(Slope)
#     gamma.append(i)
# 
# test_scatter
# p = np.where(test_scatter == np.min(test_scatter))
# P = p[0]
# test_norm[P[0]],test_slope[P[0]],gamma[P[0]],test_scatter[P[0]]
# =============================================================================

cov = np.cov(sigma_Lx,sigma_c)
yarray = log_Lx - 0.339*log_c
xarray = log_T_new
yerr = np.sqrt( (sigma_Lx)**2 + (0.339*sigma_c)**2 )- 2*0.339*cov[0][1]
xerr = sigma_T 
test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)



plt.errorbar(log_T,yarray,xerr= sigma_T,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($L_{X}/C^{0.339}$-T)' )
plt.errorbar(log_T,log_Lx,xerr= sigma_T,yerr = sigma_Lx,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($L_{X}-T$)')

z = test_Ycept+ test_Slope* log_T_new
z1 = ycept + Slope* log_T_new

plt.plot(log_T,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
plt.plot(log_T,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )

plt.xlabel('$log_{10}$(T / keV)')
plt.ylabel('$log_{10}{Y}$')
plt.title('$L_{\mathrm{X}}$/C - T scaling relation')
plt.legend(loc='best')
#plt.savefig('R-T_best_fit-NCC.png',dpi=300)
plt.show()

# Calculating BIC index
#BIC = general_functions.calculate_BIC(yarray,z,2)
# bOOTSTRAP

# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0, 10000):
#     random_clusters = cluster_total.sample(n=len(cluster_total), replace=True)
# 
#     Z = (random_clusters['z'])
#     Lx = random_clusters['Lx(1e44)']
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#     Lx_new = Lx/E
#     T = random_clusters['T(keV)']
#     T_new = T/4.5
#     log_Lx = np.log10(Lx_new)
#     log_T_new = np.log10(T_new)
#     sigma_Lx = 0.4343*random_clusters['eL(%)']/100
#     sigma_T = 0.4343*((random_clusters['Tmax']-random_clusters['Tmin'])/(2*T))
#     c = random_clusters['c']/np.median(random_clusters['c'])
#     e_c = random_clusters['e_c']
#     log_c = np.log10(c)
#     sigma_c = 0.4343 * e_c/c
#     yarray = log_Lx - 0.32*log_c
#     xarray = log_T_new
#     yerr = np.sqrt( (sigma_Lx)**2 + (0.32*sigma_c)**2 )- 2*0.32*cov[0][1]
#     xerr = sigma_T 
#     test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
# 
#     best_A.append(test_Norm)
#     best_B.append(test_Slope)
#     best_scatter.append(test_Scatter)
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict)
# bestfit_bootstrap.to_csv('Lx-T_all(c_inc)_BCES.csv')
# 
# =============================================================================




data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/Lx-T_all(c_inc)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Tcut = general_functions.calculate_asymm_err(norm)
errslope_Tcut = general_functions.calculate_asymm_err(slope)
errscatter_Tcut = general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm_Tcut,3)} +/- {np.round(errnorm_Tcut,3)}')
print(f'Slope : {np.round(Slope_Tcut,3)} +/- {np.round(errslope_Tcut,3)}')
print(f'Scatter: {np.round(Scatter_Tcut,3)} +/- {np.round(errscatter_Tcut,3)}')


# =============================================================================
# Normalization : 1.438 +/- [-0.064  0.073]
# Slope : 2.045 +/- [-0.067  0.067]
# Scatter: 0.222 +/- [-0.014  0.014]
# =============================================================================



################################################################################
                          # To constain gamma an g simultaneously
##############################################################################3##
# =============================================================================
# gamma_range = np.arange(-7.5,-5,0.01)
# g_range = np.arange(0,1,0.01)
# 
# test_scatter = []
# test_norm = []
# test_slope = []
# gamma = []
# g = []
# cov = np.cov(sigma_Lx,sigma_c)
# for i in gamma_range:
#     for j in g_range:
#         Z = (cluster_total['z'])
#         Lx = cluster_total['Lx(1e44erg/s)']
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         Lx_new = cluster_total['Lx(1e44erg/s)']*(E**i)
#         T = cluster_total['T(keV)']
#         T_new = T/4.5
#         
#         log_Lx = np.log10(Lx_new)
#         log_T = np.log10(T)
#         log_T_new = np.log10(T_new)
#         sigma_Lx = 0.4343*cluster_total['e_L(%)']/100
#         sigma_T = 0.4343*((cluster_total['Tmax']-cluster_total['Tmin'])/(2*T))
# 
#         
#         
#         c = cluster_total['c']/np.median(cluster_total['c'])
#         e_c = cluster_total['e_c']
#         log_c = np.log10(c)
#         sigma_c = 0.4343 * e_c/c
#         
#         cov = np.cov(sigma_Lx,sigma_c)
#         yarray = log_Lx - j*log_c
#         yerr = np.sqrt( (sigma_Lx)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#         xarray = log_T_new
#         xerr = sigma_T
#         
#         ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#         test_scatter.append(Scatter)
#         test_norm.append(Norm)
#         test_slope.append(Slope)
#         gamma.append(i)
#         g.append(j)
# 
# p = np.where(test_scatter == np.min(test_scatter))
# P = p[0]
# test_scatter[P[0]],gamma[P[0]],g[P[0]]
# 
# ===============================================





############## SCaling relation including the concentration parameter #####

master_file = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv', )
master_file = general_functions.cleanup(master_file)

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv',sep = ',')
thesis_table = general_functions.cleanup(thesis_table)
thesis_table.iloc[0]
cluster_total = pd.merge(master_file, thesis_table, how = 'inner', left_on='Cluster', right_on = 'Cluster' )
offset = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_FINAL_ANISOTROPY_BCG_OFFSET.csv')

cluster_total = pd.merge(cluster_total, offset, how = 'inner', left_on='Cluster', right_on = 'Cluster' )


cluster_total = cluster_total.dropna(axis=0,subset = ['c'], inplace = False)

omega_m = 0.3
omega_lambda = 0.7
Z = (cluster_total['z_x'])
Lx = cluster_total['Lx(1e44)']
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx_new = cluster_total['Lx(1e44)']/E
T = cluster_total['T(keV)_x']
T_new = T/4.5

log_Lx = np.log10(Lx_new)
log_T = np.log10(T)
log_T_new = np.log10(T_new)
sigma_Lx = 0.4343*cluster_total['eL(%)']/100
sigma_T = 0.4343*((cluster_total['Tmax_x']-cluster_total['Tmin_x'])/(2*T))

err_Lx = cluster_total['eL(%)']*Lx_new/100
err_T = [(T-cluster_total['Tmin_x']), (cluster_total['Tmax_x']-T)]
ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_Lx, sigma_Lx)

######## Defining new variables #############
# Defining the addiotional parameter as z
cluster_total.iloc[0]
metal = cluster_total['Metal(solar)']/np.median(cluster_total['Metal(solar)'])
neutral_H = cluster_total['N_Hneutral']/np.median(cluster_total['N_Hneutral'])
bcg_offset = cluster_total['BCG_offset_R500']/np.median(cluster_total['BCG_offset_R500'])
# Combining the new parameter and c together in x
x =neutral_H
log_x  = np.log(x)
c = cluster_total['c']/np.median(cluster_total['c'])
log_c = np.log(c)
e_c = cluster_total['e_c']
#e_x = (cluster_total['Met_max']+cluster_total['Met_min'])/2

log_x = np.log10(x)
#sigma_x = 0.4343 * e_x/x
sigma_x = 0
sigma_c = 0.4343* e_c/c
### To constrain the  g factor for concentration ##############
g = np.arange(-0.2,0.5,0.010)
e = np.arange(-0.3,0.5,0.01)

test_scatter = []
test_norm = []
test_slope = []
G = []
E = []
for i in g:
    for j in e:
        yarray = log_Lx - i*log_c-j*log_x
        yerr = np.sqrt( (sigma_Lx)**2 + (i*sigma_c)**2 )#+ (j*sigma_x)**2)
        xarray = log_T_new
        xerr = sigma_T
        
        ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
        test_scatter.append(Scatter)
        test_norm.append(Norm)
        test_slope.append(Slope)
        G.append(i)
        E.append(j)
    print(i)

test_scatter
p = np.where(test_scatter == np.min(test_scatter))
P = p[0]
G = G[P[0]]
E = np.round(E[P[0]],3)

yarray = log_Lx - G*log_c - E*log_x
xarray = log_T_new
yerr = np.sqrt( (sigma_Lx)**2 + (G*sigma_c)**2 + (E*sigma_x)**2)
xerr = sigma_T 
test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)



plt.errorbar(log_T,yarray,xerr= sigma_T,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation' )
plt.errorbar(log_T,log_Lx,xerr= sigma_T,yerr = sigma_Lx,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($L_{X}-T$)')

z = test_Ycept+ test_Slope* log_T_new
#z1 = ycept + Slope* log_T_new

plt.plot(log_T,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
#plt.plot(log_T,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )

plt.xlabel('$log_{10}$(T / keV)')
plt.ylabel('$log_{10}{Y}$')
plt.title('$L_{X}$/C - T scaling relation')
plt.legend(loc='best')
#plt.savefig('R-T_best_fit-NCC.png',dpi=300)
plt.show()

# BIC test
# Here we add a parameter with or without the concentration parameter 

BIC = general_functions.calculate_BIC(yarray,z,2)
BIC



# =============================================================================
# sns.set_context('paper')
# weights = np.ones_like(Lx)/len(Lx)
# plt.hist(Lx, bins=10, label = 'Normalizations a' )
# #plt.axvline(np.median(Lx), label='median', color = 'black')
# 
# plt.xlabel(r'$L_{\mathrm{X}}$ ($10^{44}$ erg/s)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,300)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Lx_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# 
# 
# sns.set_context('paper')
# weights = np.ones_like(Z)/len(Z)
# plt.hist(Z, bins=10, label = 'NormaliZations a' )
# #plt.axvline(np.median(Z), label='median', color = 'black')
# 
# plt.xlabel('$z$')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normaliZations')
# #plt.legend(loc = 'upper right')
# plt.xlim(-0.03,0.55)
# plt.ylim(0,150)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/Z_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# 
# 
# sns.set_context('paper')
# weights = np.ones_like(T)/len(T)
# plt.hist(T, bins=10, label = 'Normalizations a' )
# #plt.axvline(np.median(T), label='median', color = 'black')
# 
# plt.xlabel('T (keV)')
# plt.ylabel('No. of clusters')
# #plt.title('$L_{X}-T$ bootstrap normalizations')
# #plt.legend(loc = 'upper right')
# #plt.xlim(1.4201,1.4225)
# plt.ylim(0,120)
# plt.savefig('/home/schubham/Thesis/Thesis/Plots/T_histogram.png',dpi = 300,bbox_inches="tight")
# plt.show()
# =============================================================================


