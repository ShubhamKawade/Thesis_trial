#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:22:36 2021

@author: schubham
"""

import numpy as np
import matplotlib.pyplot as plt
import general_functions
import pandas as pd



data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/re-di_percentge_diff.csv')
data['scaling_relation']

norm_diff_obs = data['Norm']*100
slope_diff_obs = data['Slope']*100
scatter_diff_obs = data['Scatter']*100
errnorm_diff_obs = data['err_norm']*100
errslope_diff_obs = data['err_slope']*100
errscatter_diff_obs = data['err_scatter']*100
best_norm_all = data['best_norm_all']
best_slope_all = data['best_slope_all']
err_slope_all = data['err_bestslope_all']




y = ([1,2,3,4,5,6,7,8,9,10,11,12])
plt.errorbar(norm_diff_obs,y,xerr = errnorm_diff_obs,ls='',fmt='.', capsize = 3,alpha=1, elinewidth = 1.2 )
ticks = ['$L_{X}-T$','$L_{X}-T(<2keV)$','$Y_{SZ}-T$','$L_{X}-Y_{SZ}$','$L_{X}-L_{BCG}$','$Y_{SZ}-L_{BCG}$','$L_{BCG}-T$','$R-L_{X}$','$R-Y_{SZ}$','$R-L_{BCG}$','$R-T$','$R-T$(<2keV)']
plt.grid(b=True, which='major', axis='y', linestyle='--')
plt.yticks(y,ticks,va='top',rotation = 'horizontal')
plt.xlabel('$(a_{R}-a_{D}) / a_{All}$')
plt.title('Normalization(a) difference in Relaxed-Disturbed')
plt.axvline(0, color='black')
#plt.xlim(-40,120)
#plt.savefig('frac-diff-Norm_all-scatter_relations',dpi=300)
plt.show()




slope_Lx_T = best_slope_all[0]
slope_Ysz_T = best_slope_all[2]
slope_Lx_Ysz = best_slope_all[3]
slope_R_Lx = best_slope_all[7]
slope_R_T = best_slope_all[10]
slope_R_Ysz = best_slope_all[8]

errslope_Lx_Lbcg = err_slope_all[0]
errslope_Ysz_Lbcg = err_slope_all[2]
errslope_Lx_Ysz = err_slope_all[3]
errslope_R_Lx = err_slope_all[7]
errslope_R_T = err_slope_all[10]
errslope_R_Ysz = err_slope_all[8]


percent_change_range_for_i = np.arange(0,200,5)
percent_change_range = np.arange(-100,210,5)

Norm_diff_exp_Lx_T = []
errNorm_diff_Lx_T = []

Norm_diff_exp_Ysz_T = []
errNorm_diff_Ysz_T = []

Norm_diff_exp_Lx_Ysz = []
errNorm_diff_Lx_Ysz = []

Norm_diff_exp_R_Lx = []
errNorm_diff_R_Lx = []

Norm_diff_exp_R_Ysz = []
errNorm_diff_R_Ysz = []

Norm_diff_exp_R_T = []
errNorm_diff_R_T = []

index_accepted_diff = []
chisq_red_all = []
I=[]
J=[]
K=[]
L = []
## i,j,k and l are used for Lx, T, Ysz and R respectively
for i in percent_change_range_for_i:
    for j in percent_change_range:
        for k in percent_change_range:
            for l in percent_change_range:
                del_Lx = i/100
                del_T = 1+j/100
                del_T_tot = (del_T ** slope_Lx_T) - 1 
                norm_diff_exp_Lx_T = (del_Lx - del_T_tot) * 100
                Norm_diff_exp_Lx_T.append(norm_diff_exp_Lx_T)
                errnorm_diff_exp_Lx_T = np.abs(del_T_tot) * np.log(slope_Lx_T) * errslope_Lx_T
                errNorm_diff_Lx_T.append(errnorm_diff_exp_Lx_T)
                
                del_Ysz = k/100
                del_T = 1+j/100
                del_T_tot = (del_T ** slope_Ysz_T)-1
                norm_diff_exp_Ysz_T = (del_Ysz - del_T_tot) * 100
                Norm_diff_exp_Ysz_T.append(norm_diff_exp_Ysz_T)
                errnorm_diff_exp_Ysz_T = np.abs(del_T_tot) * np.log(slope_Ysz_T) * errslope_Ysz_T
                errNorm_diff_Ysz_T.append(errnorm_diff_exp_Ysz_T)
                
                del_Lx = i/100
                del_Ysz = 1+k/100
                del_Ysz_tot =( del_Ysz ** slope_Lx_Ysz) - 1
                norm_diff_exp_Lx_Ysz = (del_Lx - del_Ysz_tot) * 100
                Norm_diff_exp_Lx_Ysz.append(norm_diff_exp_Lx_Ysz)
                errnorm_diff_exp_Lx_Ysz = np.abs(del_T_tot) * np.log(slope_Lx_Ysz) * errslope_Lx_Ysz
                errNorm_diff_Lx_Ysz.append(errnorm_diff_exp_Lx_Ysz)
                
                del_R = l/100
                del_Lx = 1+i/100
                del_Lx_tot = (del_Lx ** slope_R_Lx) - 1
                norm_diff_exp_R_Lx = (del_R - del_Lx_tot) * 100
                Norm_diff_exp_R_Lx.append(norm_diff_exp_R_Lx)
                errnorm_diff_exp_R_Lx = np.abs(del_T_tot) * np.log(slope_R_Lx) * errslope_R_Lx
                errNorm_diff_R_Lx.append(errnorm_diff_exp_R_Lx)
                
                del_R = l/100
                del_T = 1+j/100
                del_T_tot = (del_T ** slope_R_T) - 1
                norm_diff_exp_R_T = (del_R - del_T_tot) * 100
                Norm_diff_exp_R_T.append(norm_diff_exp_R_T)
                errnorm_diff_exp_R_T = np.abs(del_T_tot) * np.log(slope_R_T) * errslope_R_T
                errNorm_diff_R_T.append(errnorm_diff_exp_R_T)
                
                del_R = l/100
                del_Ysz = 1+k/100
                del_Ysz_tot = (del_Ysz ** slope_R_Ysz) - 1
                norm_diff_exp_R_Ysz = (del_R - del_Ysz_tot) * 100
                Norm_diff_exp_R_Ysz.append(norm_diff_exp_R_Ysz)
                errnorm_diff_exp_R_Ysz = np.abs(del_T_tot) * np.log(slope_R_Ysz) * errslope_R_Ysz
                errNorm_diff_R_Ysz.append(errnorm_diff_exp_R_Ysz)
                
                data = np.array([norm_diff_exp_Lx_T, norm_diff_exp_Ysz_T, norm_diff_exp_Lx_Ysz,norm_diff_exp_R_Lx,norm_diff_exp_R_T,norm_diff_exp_R_Ysz])
                model = np.array([norm_diff_obs[0],norm_diff_obs[2],norm_diff_obs[3],norm_diff_obs[7],norm_diff_obs[10],norm_diff_obs[8]])
                err = np.array([errnorm_diff_obs[0],errnorm_diff_obs[2],errnorm_diff_obs[3],errnorm_diff_obs[7],errnorm_diff_obs[10],errnorm_diff_obs[8]])
                chisq = np.sum(((data-model)/err)**2)
                chisq_red = chisq/ (2) 
                if chisq_red < 7:
                    chisq_red_all.append(chisq_red)

                    I.append(i)
                    J.append(j)
                    K.append(k)
                    L.append(l)

    print(i)
bestfit_bootstrap_dict = {'Lx': I, 'Ysz': K, 'T': J, 'R' : L}              
bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
bestfit_bootstrap.to_csv('percent_diff_accepted_1.csv')
plt.hist(I,bins=20)
plt.show()


data = pd.read_csv('/home/schubham/Downloads/percent_diff_accepted_rVd(step_5).csv')
print(np.min(data['Lx'])),print(np.max(data['Lx']))
print(np.min(data['T'])),print(np.max(data['T']))
print(np.min(data['Ysz'])),print(np.max(data['Ysz']))
print(np.min(data['R'])),print(np.max(data['R']))
print(np.min(data['Lbcg'])),print(np.max(data['Lbcg']))
