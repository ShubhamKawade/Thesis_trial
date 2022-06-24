#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:47:50 2021

@author: schubham
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 

data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/percent_diff_accepted_ccVncc_all(step_5_chsqlim_3).csv')
data.iloc[0]
del_Lx = data['Lx']
del_Ysz = data['Ysz']
del_T = data['T']
del_R = data['R']
del_Lbcg = data['Lbcg']
print(np.min(del_Lx), np.max(del_Lx))
print(np.min(del_T), np.max(del_T))

print(np.min(del_Ysz), np.max(del_Ysz))
print(np.min(del_R), np.max(del_R))
print(np.min(del_Lbcg), np.max(del_Lbcg))

red_chisq = data['chisq_red']
Lx_T_accepted =  data['Lx_T_accepted']
Ysz_T_accepted =  data['Ysz_T_accepted']
Lx_Ysz_accepted =   data['Lx_Ysz_accepted']
Lx_Lbcg_accepted =    data['Lx_Lbcg_accepted']
Ysz_Lbcg_accepted =  data['Ysz_Lbcg_accepted']
Lbcg_T_accepted = data['Lbcg_T_accepted']
R_Lx_accepted =     data['R_Lx_accepted']
R_T_accepted =      data['R_T_accepted']
R_Ysz_accepted =    data['R_Ysz_accepted']
R_Lbcg_accepted = data['R_Lbcg_accepted']

err_Lx_T_accepted =  data['err_Lx_T_accepted']
err_Ysz_T_accepted =  data['err_Ysz_T_accepted']
err_Lx_Ysz_accepted =   data['err_Lx_Ysz_accepted']
err_Lx_Lbcg_accepted =    data['err_Lx_Lbcg_accepted']
err_Ysz_Lbcg_accepted =  data['err_Ysz_Lbcg_accepted']
err_Lbcg_T_accepted = data['err_Lbcg_T_accepted']
err_R_Lx_accepted =     data['err_R_Lx_accepted']
err_R_T_accepted =      data['err_R_T_accepted']
err_R_Ysz_accepted =    data['err_R_Ysz_accepted']
err_R_Lbcg_accepted = data['err_R_Lbcg_accepted']

data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/ccVncc_percent_diff_(all+clusters)_10.csv')
data['scaling_relation']

norm_diff_obs = data['Norm_all']*100
slope_diff_obs = data['Slope_all']*100
scatter_diff_obs = data['Scatter_all']*100
errnorm_diff_obs = data['err_Norm_all']*100
errslope_diff_obs = data['err_Slope_all']*100
errscatter_diff_obs = data['err_Scatter_all']*100
best_norm_all = data['best_Norm_all']
best_slope_all = data['best_Slope_all']
err_slope_all = data['err_bestslope_all']


# To get value corresp[0]onding to the lowest red_chisq
p = np.where(red_chisq == np.min(red_chisq))
P = p[0]
#lowest red_chisq at index p[0]
red_chisq[P[0]]
err_Lx_T_diff_exp = err_Lx_T_accepted[P[0]] 
err_Ysz_T_diff_exp = err_Ysz_T_accepted[P[0]]
err_Lx_Ysz_diff_exp = err_Lx_Ysz_accepted[P[0]]
err_Lx_Lbcg_diff_exp = err_Lx_Lbcg_accepted[P[0]] 
err_Ysz_Lbcg_diff_exp = err_Ysz_Lbcg_accepted[P[0]] 
err_Lbcg_T_diff_exp = err_Lbcg_T_accepted[P[0]]
err_R_Lx_diff_exp = err_R_Lx_accepted[P[0]] 
err_R_T_diff_exp = err_R_T_accepted[P[0]] 
err_R_Ysz_diff_exp = err_R_Ysz_accepted[P[0]] 
err_R_Lbcg_diff_exp = err_R_Lbcg_accepted[P[0]] 

Lx_T_diff_exp = Lx_T_accepted[P[0]] 
Ysz_T_diff_exp =Ysz_T_accepted[P[0]]
Lx_Ysz_diff_exp = Lx_Ysz_accepted[P[0]]
Lx_Lbcg_diff_exp = Lx_Lbcg_accepted[P[0]] 
Ysz_Lbcg_diff_exp = Ysz_Lbcg_accepted[P[0]] 
Lbcg_T_diff_exp = Lbcg_T_accepted[P[0]]
R_Lx_diff_exp = R_Lx_accepted[P[0]] 
R_T_diff_exp = R_T_accepted[P[0]] 
R_Ysz_diff_exp = R_Ysz_accepted[P[0]] 
R_Lbcg_diff_exp = R_Lbcg_accepted[P[0]] 

norm_diff_exp = [Lx_T_diff_exp, Ysz_T_diff_exp, Lx_Ysz_diff_exp, Lx_Lbcg_diff_exp, Ysz_Lbcg_diff_exp, Lbcg_T_diff_exp, R_Lx_diff_exp, R_T_diff_exp, R_Ysz_diff_exp, R_Lbcg_diff_exp]
err_norm_diff_exp = [err_Lx_T_diff_exp, err_Ysz_T_diff_exp, err_Lx_Ysz_diff_exp, err_Lx_Lbcg_diff_exp, err_Ysz_Lbcg_diff_exp,err_Lbcg_T_diff_exp, err_R_Lx_diff_exp, err_R_T_diff_exp, err_R_Ysz_diff_exp, err_R_Lbcg_diff_exp]

del_Lx_exp = del_Lx[P[0]] 
del_Ysz_exp = del_Ysz[P[0]] 
del_T_exp = del_T[P[0]] 
del_R_exp = del_R[P[0]] 
del_Lbcg_exp  = del_Lbcg[P[0]] 


y = ([1,2,3,4,5,6,7,8,9,10])
plt.errorbar(norm_diff_exp,y,xerr = err_norm_diff_exp,ls='',fmt='.', capsize = 3,alpha=1, elinewidth = 1.2,label = 'Expected (Best fit)' )
plt.errorbar(norm_diff_obs,y,xerr = errnorm_diff_obs,ls='',fmt='.', capsize = 3,alpha=1, elinewidth = 1.2, label = 'Observed' )
ticks = ['$L_{X}-T$','$Y_{SZ}-T$','$L_{X}-Y_{SZ}$','$L_{X}-L_{BCG}$','$Y_{SZ}-L_{BCG}$','$L_{BCG}-T$','$R-L_{X}$','$R-T$','$R-Y_{SZ}$','$R-L_{BCG}$']
plt.grid(b=True, which='major', axis='y', linestyle='--')
plt.yticks(y,ticks,va='top',rotation = 'horizontal')
plt.xlabel('$(a_{CC}-a_{NCC}) / a_{All}$ [%]')
plt.title('Norm difference in CC-NCC (All)(step : 5%, $\chi^{2}_{red}$ lim : 3)')
plt.axvline(0, color='black')
plt.legend()
plt.xlim(-100,140)

#plt.xlim(-40,120)
#plt.savefig('frac-dif/home/schubham/Thesis/Thesis/Data/percent_diff_accepted_ccVncc(step_3,chisq_lim_3)_test.csvf-Norm_all-scatter_relations',dpi=300)
plt.show()


general_functions.calculate_asymm_err(del_Lbcg)

bins=10
# =============================================================================
# def plot_hist(array, bins=bins,weights=weights):
#     return plt.hist(array, bins=bins,weights=weights)
# 
# =============================================================================
weights = np.ones_like(del_Lx)/len(del_Lx)
plt.hist(del_Lx,bins=bins,weights=weights, alpha = 0.6)
plt.axvline(np.median(del_Lx), color ='blue', label = f'median ({np.median(del_Lx)}%)')
plt.axvline(del_Lx_exp, color ='red', label = f'Best fit ({del_Lx_exp}%)')
plt.title('$\Delta$Lx for $\chi^2_{red}$ < 7')
plt.legend()
plt.xlabel('$\Delta$Lx %')
plt.show()

weights = np.ones_like(del_T)/len(del_T)
plt.hist(del_T,bins=bins,weights=weights, alpha = 0.6)
plt.axvline(np.median(del_T), color ='blue', label = f'median ({np.median(del_T)}%)')
plt.axvline(del_T_exp, color ='red', label = f'Best fit ({del_T_exp}%)')
plt.title('$\Delta$T for $\chi^2_{red}$ < 7')
plt.legend()
plt.xlabel('$\Delta$T %')
plt.show()

weights = np.ones_like(del_Ysz)/len(del_Ysz)
plt.hist(del_Ysz,bins=bins,weights=weights, alpha = 0.6)
plt.axvline(np.median(del_Ysz), color ='blue', label = f'median ({np.median(del_Ysz)}%)')
plt.axvline(del_Ysz_exp, color ='red', label = f'Best fit ({del_Ysz_exp}%)')
plt.title('$\Delta$Ysz for $\chi^2_{red}$ < 7')
plt.legend()
plt.xlabel('$\Delta$Ysz %')
plt.show()

weights = np.ones_like(del_R)/len(del_R)
plt.hist(del_R,bins=8,weights=weights, alpha = 0.6)
plt.axvline(np.median(del_R), color ='blue', label = f'median ({np.median(del_R)}%)')
plt.axvline(del_R_exp, color ='red', label = f'Best fit ({del_R_exp}%)')
plt.title('$\Delta$R for $\chi^2_{red}$ < 7')
plt.legend()
plt.xlabel('$\Delta$R %')
plt.show()

weights = np.ones_like(del_Lbcg)/len(del_Lbcg)
plt.hist(del_Lbcg,bins=bins,weights=weights, alpha = 0.6)
plt.axvline(np.median(del_Lbcg), color ='blue', label = f'median ({np.median(del_Lbcg)}%)')
plt.axvline(del_Lbcg_exp, color ='red', label = f'Best fit ({del_Lbcg_exp}%)')
plt.title('$\Delta$Lbcg for $\chi^2_{red}$ < 7')
plt.legend()
plt.xlabel('$\Delta$Lbcg %')
plt.show()

