#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 
import seaborn as sns


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


# Constraining gamma
# =============================================================================
# re_range = np.arange(-5,5,0.01)
# norm = []
# scatter = []
# slope = []
# re = []
# 
# ## A for loop t run throught each value of re_range and getting Norm, slope and scatter values.
# for i in re_range:
#     
#     r_new = (R_kpc/250) * (E**(i))
#     log_r = np.log10(r_new)
#     T = rt['T']
#     log_T = np.log10(T)
#     log_T_new = np.log10(T/4.5)
#     sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
#     
#     ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_r, sigma_r)
#     norm.append(Norm)
#     slope.append(Slope)
#     scatter.append(Scatter)
#     re.append(i)
#     
# # To return the index which corresponds to the minimum scatter    
# p = np.where(scatter == np.min(scatter))
# P = p[0]
# re[P[0]]
# =============================================================================
# gamma = 1.89
# Uncertainity on gamma
# =============================================================================
# RE = []
# 
# for j in range(1000):
# 
#     random_clusters = rt.sample(n = len(rt), replace = True)
#     re_range = np.arange(-7,13,0.1)
#     scatter = []
#     re = []
# 
#     ## A for loop t run throught each value of re_range and getting Norm, slope and scatter values.
#     for i in re_range:
#         
#         random_clusters = rt.sample(n = len(rt), replace = True)
#         Z = (random_clusters['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         R = random_clusters['R kpc']
#         R_min = random_clusters['Rmin kpc']
#         R_max = random_clusters['Rmax kpc']
# 
#         r_new = (R_kpc/250) * (E**(i))
#         log_r = np.log10(r_new)
#         T = rt['T']
#         log_T = np.log10(T)
#         log_T_new = np.log10(T/4.5)
#         sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
#         
#         ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(log_T_new, sigma_T, log_r, sigma_r)
#         
#         scatter.append(Scatter)
#         re.append(i)
# 
#     ## To return the index which corresponds to the minimum scatter    
#     p = np.where(scatter == np.min(scatter))
#     P = p[0]
#     RE.append(re[P[0]])
#     print(j)
# =============================================================================

# =============================================================================
# plt.hist(RE,bins=20)
# weights = np.ones_like(RE)/len(RE)
# plt.hist(RE,bins=20, weights = weights, alpha = 0.7)
# plt.xlabel('$\gamma_{ R-T}$')
# plt.ylabel('Count')
# plt.title('Bootstrap for $\gamma_{ R-T}$')
# plt.axvline(np.mean(RE), ls='-', color = 'black')
# np.max(RE),np.min(RE)
# 
# general_functions.calculate_asymm_err(RE)
# =============================================================================
R = rt['R kpc']
R_min = rt['Rmin kpc']
R_max = rt['Rmax kpc']
R_new = (R/250) * (E**(1.89))
log_r = np.log10(R_new)
err_r = [R-R_min,R_max-R]

sigma_r = 0.4343 * (Rmax_kpc-Rmin_kpc)/(2*R_kpc)


T = rt['T']
T_new = T/4.5
log_T = np.log(T)
log_T_new = np.log10(T/4.5)
sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
err_T = [T-rt['Tmin'], rt['Tmax']-T]
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)

sns.set_context('paper')

plt.errorbar(T, R, yerr=err_r, xerr=err_T, ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_T)})')
T_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept, Slope)
plt.plot(T_linspace,z,label='Best fit', color ='green')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(' T (keV)')
plt.ylabel(' $R*E(z)^{1.89}$ (kpc)')
plt.title('$R-T$ best fit')
plt.legend(loc = 'lower right')
plt.xlim(0.3,50)
plt.ylim(10,3000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-T_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()

# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = rt.sample(n = len(rt), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
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
#     print(j)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_all_BCES.csv') 
# =============================================================================


data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_all_BCES.csv')

norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']


errnorm = general_functions.calculate_asymm_err(norm)
errslope = general_functions.calculate_asymm_err(slope)
errscatter = general_functions.calculate_asymm_err(scatter)
print('The best fit parameters are :')
print(f'Normalization : {np.round(Norm,3)} +/- {np.round(errnorm,3)}')
print(f'Slope : {np.round(Slope,3)} +/- {np.round(errslope,3)}')
print(f'Scatter: {np.round(Scatter,3)} +/- {np.round(errscatter,3)}')



#####################################################################

         # Cutting galaxy groups based on mass 

###########################################################################################
rt = general_functions.removing_galaxy_groups(rt)
Z = (rt['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
R = rt['R kpc']
R_min = rt['Rmin kpc']
R_max = rt['Rmax kpc']
R_new = (R/250) * (E**(1.89))
log_r = np.log10(R_new)
err_r = [R-R_min,R_max-R]

sigma_r = 0.4343 * (Rmax_kpc-Rmin_kpc)/(2*R_kpc)


T = rt['T']
T_new = T/4.5
log_T = np.log(T)
log_T_new = np.log10(T/4.5)
sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
err_T = [T-rt['Tmin'], rt['Tmax']-T]
ycept_Mcut,Norm_Mcut,Slope_Mcut,Scatter_Mcut = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)

sns.set_context('paper')

plt.errorbar(T, R, yerr=err_r, xerr=err_T, ls='',fmt='.', capsize = 1.7,alpha=0.8, elinewidth = 0.65 ,color='red', label=f'Clusters ({len(log_T)})')
T_linspace = np.linspace(0.0001,3000,100)
z = general_functions.plot_bestfit(T_linspace, 4.5, 250, ycept_Mcut, Slope_Mcut)
plt.plot(T_linspace,z,label='Best fit', color ='green')
plt.xscale('log')
plt.yscale('log')
#plt.errorbar(log_Lx,log_r,sigma_Lx,sigma_r,ls='none')
plt.xlabel(' T (keV)')
plt.ylabel(' $R*E(z)^{1.89}$ (kpc)')
plt.title('$R-T$ best fit ($M_{\mathrm{cluster}} > 10^{14}\mathrm{M}_{\odot}$)')
plt.legend(loc = 'lower right')
plt.xlim(0.3,50)
plt.ylim(10,3000)
plt.savefig('/home/schubham/Thesis/Thesis/Plots/Best fit/R-T_Mcut_bestfit.png',dpi=300,bbox_inches="tight")
plt.show()



# Bootstrap
# =============================================================================
# best_A = []
# best_B = []
# best_scatter = []
# for j in range(0,10000):
#     
#     random_clusters = rt.sample(n = len(rt), replace = True)
#     Z = (random_clusters['z'])
#     E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# 
#     R = random_clusters['R kpc']
#     R_min = random_clusters['Rmin kpc']
#     R_max = random_clusters['Rmax kpc']
#     R_new = (R/250) * (E**(1.89))
#     log_r = np.log10(R_new)
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
#     print(j)
# 
# bestfit_bootstrap_dict = {'Normalization': best_A, 'Slope': best_B, 'Scatter': best_scatter}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_all(Mcut)_BCES.csv')
# =============================================================================



data = pd.read_csv('/home/schubham/Thesis/Thesis/Scaling_relations/bces_data/R-T_all(Mcut)_BCES.csv')
norm = data['Normalization']
slope = data['Slope']
scatter = data['Scatter']

errnorm_Mcut = general_functions.calculate_asymm_err(norm)
errslope_Mcut = general_functions.calculate_asymm_err(slope)
errscatter_Mcut = general_functions.calculate_asymm_err(scatter)

print(f'Normalization : {np.round(Norm_Mcut,3)} +/- {np.round(errnorm_Mcut,3)}')
print(f'Slope : {np.round(Slope_Mcut,3)} +/- {np.round(errslope_Mcut,3)}')
print(f'Scatter: {np.round(Scatter_Mcut,3)} +/- {np.round(errscatter_Mcut,3)}')





















####################################################################################3

r = pd.read_fwf('/home/schubham/Thesis/Thesis/Data/Half-radii-T-NEW-2.txt',sep ='\\s+')
r.rename({'#Name':'Cluster'},axis=1,inplace=True)
r = general_functions.cleanup(r)
rt = r[r['R'] > 2]

R_old = rt['R']
R_new = general_functions.correct_psf(R_old)
Rmin_old = rt['R'] - rt['Rmin']
Rmax_old = rt['R'] + rt['Rmax']
Rmax_new = general_functions.correct_psf(Rmax_old)
Rmin_new = general_functions.correct_psf(Rmin_old)


Z = (rt['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
d_A = cosmo.angular_diameter_distance(Z)*1000

theta = (R_new/60)*np.pi/180
R_kpc = theta * d_A.value

theta_min = (Rmin_new/60)*np.pi/180
Rmin_kpc = theta_min * d_A.value

theta_max = (Rmax_new/60)*np.pi/180
Rmax_kpc = theta_max * d_A.value
    
rt['R kpc'] = R_kpc
rt['Rmin kpc'] = Rmin_kpc
rt['Rmax kpc'] = Rmax_kpc

thesis_table = pd.read_csv('/home/schubham/Thesis/Thesis/Data/thesis_table.csv')
thesis_table = thesis_table.rename({'cluster_name':'Cluster'},axis=1)
thesis_table = general_functions.cleanup(thesis_table)
rt = pd.merge(rt, thesis_table, right_on='Cluster',left_on = 'Cluster', how ='inner')

Z = (rt['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5

R = rt['R kpc']
R_min = rt['Rmin kpc']
R_max = rt['Rmax kpc']
R_new = (R/250) * (E**(1.9))
log_r = np.log10(R_new)
err_r = [(R-R_min)/250,(R_max-R)/250]

sigma_r = 0.4343 * (R_max-R_min)/(2*R)


T = rt['T']
T_new = T/4.5
log_T = np.log(T)
log_T_new = np.log10(T/4.5)
sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
err_T = [T-rt['Tmin'], rt['Tmax']-T]
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(log_T_new,sigma_T,log_r,sigma_r)

############ Adding new parameter for the new scaling relation ############

c = rt['c']/np.median(rt['c'])
e_c = rt['e_c']
log_c = np.log10(c)
sigma_c = 0.4343 * e_c/c
cov = np.cov(sigma_r,sigma_c)

g = np.arange(-3,3,0.01)
test_scatter = []
test_norm = []
test_slope = []
gamma = []
cov = np.cov(sigma_r,sigma_c)
for i in g:
    yarray = log_r - i*log_c
    yerr = np.sqrt( (sigma_r)**2 + (i*sigma_c)**2 - 2*i*cov[0][1])
    xarray = log_T_new
    xerr = sigma_T
    
    ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
    test_scatter.append(Scatter)
    test_norm.append(Norm)
    test_slope.append(Slope)
    gamma.append(i)

test_scatter
p = np.where(test_scatter == np.min(test_scatter))
P = p[0]
test_norm[P[0]],test_slope[P[0]],gamma[P[0]],test_scatter[P[0]]

yarray = log_r - (-0.39)*log_c
xarray = log_T_new
yerr = np.sqrt( (sigma_r)**2 + (-0.39*sigma_c)**2 + 2*-0.39*cov[0][1] )
xerr = sigma_T
test_Ycept, test_Norm, test_Slope, test_Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)



plt.errorbar(log_T,yarray,xerr= xerr,yerr = yerr,color = 'green',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'new relation ($R/C^{-0.39}$-T)' )
plt.errorbar(log_T,log_r,xerr= sigma_T,yerr = sigma_r,color = 'red',ls='',fmt=' ', capsize = 1,alpha= 1, elinewidth = 1, label = 'old relation ($R-T$)')

z = test_Ycept+ test_Slope* log_T_new
z1 = ycept + Slope* log_T_new

plt.plot(log_T,z, color = 'blue',label = f'New bestfit ($\sigma$ = {np.round(test_Scatter,3)})')
plt.plot(log_T,z1, color = 'black',label = f'old bestfit($\sigma$ = {np.round(Scatter,3)})' )

plt.xlabel('log(T)')
plt.ylabel('$log{r}$')
plt.title('$R/c - T}$scaling relation')
plt.legend(loc='best')
#plt.savefig('R-T_best_fit-NCC.png',dpi=300)
plt.show()


################################################################################
                          # To constain gamma an g simultaneously
##############################################################################3##
re_range = np.arange(-1,3,0.01)
g_range = np.arange(-1,1,0.01)

# =============================================================================
# test_scatter = []
# test_norm = []
# test_slope = []
# gamma = []
# g = []
# cov = np.cov(sigma_r,sigma_c)
# for i in re_range:
#     for j in g_range:
#         
#         Z = (rt['z'])
#         E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#         
#         R = rt['R kpc']
#         R_min = rt['Rmin kpc']
#         R_max = rt['Rmax kpc']
#         R_new = (R/250) * (E**(i))
#         log_r = np.log10(R_new)
#         sigma_r = 0.4343 * (R_max-R_min)/(2*R)
#         
#         T = rt['T']
#         T_new = T/4.5
#         log_T = np.log(T)
#         log_T_new = np.log10(T/4.5)
#         sigma_T = 0.4343 * (rt['Tmax']-rt['Tmin'])/(2*T)
#         
#         c = rt['c']/np.median(rt['c'])
#         e_c = rt['e_c']
#         log_c = np.log10(c)
#         sigma_c = 0.4343 * e_c/c
#         cov = np.cov(sigma_r,sigma_c)
#         
#         yarray = log_r - j*log_c
#         yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#         xarray = log_T_new
#         xerr = sigma_T
#         
#         ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#         test_scatter.append(Scatter)
#         test_norm.append(Norm)
#         test_slope.append(Slope)
# 
#       
#         gamma.append(i)
#         g.append(j)
# 
# p = np.where(test_scatter == np.min(test_scatter))
# P = p[0]
# test_scatter[P[0]],gamma[P[0]],g[P[0]]
# 
# =============================================================================

# Botstrap for gamma and g
# =============================================================================
# re_range = np.arange(-0.15,2.5,0.01)
# g_range = np.arange(-0.6,-0.2,0.01)
# 
# test_scatter = []
# gamma_bootstrap = []
# g_bootstrap = []
# gamma = []
# g = []
# for k in range(2):
#     random_clusters = rt.sample(n = len(rt), replace = True)
#     
#     for i in re_range:
#         for j in g_range:
#             Z = (random_clusters['z'])
#             E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
#             
#             R = random_clusters['R kpc']
#             R_min = random_clusters['Rmin kpc']
#             R_max = random_clusters['Rmax kpc']
#             R_new = (R/250) * (E**(i))
#             log_r = np.log10(R_new)
#             sigma_r = 0.4343 * (R_max-R_min)/(2*R)
#             
#             T = random_clusters['T']
#             T_new = T/4.5
#             log_T = np.log(T)
#             log_T_new = np.log10(T/4.5)
#             sigma_T = 0.4343 * (random_clusters['Tmax']-random_clusters['Tmin'])/(2*T)
#             
#             c = random_clusters['c']/np.median(random_clusters['c'])
#             e_c = random_clusters['e_c']
#             log_c = np.log10(c)
#             sigma_c = 0.4343 * e_c/c
#             cov = np.cov(sigma_r,sigma_c)
#             
#             yarray = log_r - j*log_c
#             yerr = np.sqrt( (sigma_r)**2 + (j*sigma_c)**2 - 2*j*cov[0][1])
#             xarray = log_T_new
#             xerr = sigma_T
#             
#             ycept, Norm, Slope, Scatter = general_functions.calculate_bestfit(xarray,xerr,yarray, yerr)
#             test_scatter.append(Scatter)
#             gamma.append(i)
#             g.append(j)
#     print(k)
#     p = np.where(test_scatter == np.min(test_scatter))
#     P = p[0]
#     gamma_bootstrap.append(gamma[P[0]])
#     g_bootstrap.append(g[P[0]])
# bestfit_bootstrap_dict = {'gamma': gamma_bootstrap, 'g': g_bootstrap}              
# bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
# bestfit_bootstrap.to_csv('R-T_gamma&_bootstrap.csv')
# 
# =============================================================================
data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/R-T_gamma&_bootstrap.csv')
gamma = data['gamma']
g = data['g']

errgamma = general_functions.calculate_asymm_err(gamma)
errg = general_functions.calculate_asymm_err(g)
plt.hist(g,bins=5)