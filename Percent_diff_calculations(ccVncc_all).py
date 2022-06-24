import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Percentage diff analysis for clusters (groups removed)

data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/ccVncc_percent_diff_(all+clusters).csv')
data['scaling_relation']

norm_diff_obs = data['Norm_clusters']*100
slope_diff_obs = data['Slope_clusters']*100
scatter_diff_obs = data['Scatter_clusters']*100
errnorm_diff_obs = data['err_Norm_clusters']*100
errslope_diff_obs = data['err_Slope_clusters']*100
errscatter_diff_obs = data['err_Scatter_clusters']*100
best_norm_Clusters = data['best_Norm_clusters']
best_slope_Clusters = data['best_Slope_clusters']
err_slope_Clusters = data['err_bestslope_clusters']




# =============================================================================
# y = ([1,2,3,4,5,6,7,8,9,10,11,12])
# plt.errorbar(norm_diff_obs,y,xerr = errnorm_diff_obs,ls='',fmt='.', capsize = 3,alpha=1, elinewidth = 1.2 )
# ticks = ['$L_{X}-T$','$L_{X}-T(<2keV)$','$Y_{SZ}-T$','$L_{X}-Y_{SZ}$','$L_{X}-L_{BCG}$','$Y_{SZ}-L_{BCG}$','$L_{BCG}-T$','$R-L_{X}$','$R-Y_{SZ}$','$R-L_{BCG}$','$R-T$','$R-T$(<2keV)']
# plt.grid(b=True, which='major', axis='y', linestyle='--')
# plt.yticks(y,ticks,va='top',rotation = 'horizontal')
# plt.xlabel('$(a_{CC}-a_{NCC}) / a_{All}$')
# plt.title('Normalization(a) difference in CC-NCC')
# plt.axvline(0, color='black')
# #plt.xlim(-40,120)
# #plt.savefig('frac-diff-Norm_all-scatter_relations',dpi=300)
# plt.show()
# 
# 
# 
# =============================================================================
slope_Lx_T = best_slope_Clusters[0]
slope_Ysz_T = best_slope_Clusters[1]
slope_Lx_Ysz = best_slope_Clusters[2]
slope_Lx_Lbcg = best_slope_Clusters[3]
slope_Ysz_Lbcg = best_slope_Clusters[4]
slope_Lbcg_T = best_slope_Clusters[5]
slope_R_Lx = best_slope_Clusters[6]
slope_R_Ysz = best_slope_Clusters[7]
slope_R_Lbcg = best_slope_Clusters[8]
slope_R_T = best_slope_Clusters[9]


errslope_Lx_T = err_slope_Clusters[0]
errslope_Ysz_T = err_slope_Clusters[1]
errslope_Lx_Ysz = err_slope_Clusters[2]
errslope_Lx_Lbcg = err_slope_Clusters[3]
errslope_Ysz_Lbcg = err_slope_Clusters[4]
errslope_Lbcg_T = err_slope_Clusters[5]
errslope_R_Lx = err_slope_Clusters[6]
errslope_R_Ysz = err_slope_Clusters[7]
errslope_R_Lbcg = err_slope_Clusters[8]
errslope_R_T = err_slope_Clusters[9]


step = 5
del_Lx_range = np.arange(0,160,step )
del_T_range = np.arange(-70,30,step )
del_Ysz_range = np.arange(-90,75,step )
del_R_range = np.arange(-90,-20,step )
del_Lbcg_range = np.arange(-65,70,step )

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

Norm_diff_exp_Lx_Lbcg = []
errNorm_diff_Lx_Lbcg = []

Norm_diff_exp_Ysz_Lbcg = []
errNorm_diff_Ysz_Lbcg = []

Norm_diff_exp_Lbcg_T = []
errNorm_diff_Lbcg_T = []

Norm_diff_exp_R_Lbcg = []
errNorm_diff_R_Lbcg = []

index_accepted_diff = []
chisq_red_all = []
I=[]
J=[]
K=[]
L = []
M=[]

Lx_T_accepted = []
Ysz_T_accepted = []
Lx_Ysz_accepted = []
Lx_Lbcg_accepted = []
Ysz_Lbcg_accepted = []
Lbcg_T_accepted = []
R_Lx_accepted = []
R_T_accepted = []
R_Ysz_accepted = []
R_Lbcg_accepted = []

err_Lx_T_accepted = []
err_Ysz_T_accepted = []
err_Lx_Ysz_accepted = []
err_Lx_Lbcg_accepted = []
err_Ysz_Lbcg_accepted = []
err_Lbcg_T_accepted = []
err_R_Lx_accepted = []
err_R_T_accepted = []
err_R_Ysz_accepted = []
err_R_Lbcg_accepted = []

## i,j,k and l are used for Lx, T, Ysz and R respectively
for i in del_Lx_range:
    for j in del_T_range:
        for k in del_Ysz_range:
            for l in del_R_range:
                for m in del_Lbcg_range:
                    del_Lx = i/100
                    del_T = 1+j/100
                    del_T_tot = (del_T ** slope_Lx_T) - 1 
                    norm_diff_exp_Lx_T = (del_Lx - del_T_tot) * 100
                    Norm_diff_exp_Lx_T.append(norm_diff_exp_Lx_T)
                    #errnorm_diff_exp_Lx_T = np.abs(del_T_tot*100) * np.log(slope_Lx_T) * errslope_Lx_T
                    errnorm_diff_exp_Lx_T = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Lx_T.append(errnorm_diff_exp_Lx_T)
                    
                    del_Ysz = k/100
                    del_T = 1+j/100
                    del_T_tot = (del_T ** slope_Ysz_T)-1
                    norm_diff_exp_Ysz_T = (del_Ysz - del_T_tot) * 100
                    Norm_diff_exp_Ysz_T.append(norm_diff_exp_Ysz_T)
                    errnorm_diff_exp_Ysz_T = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Ysz_T.append(errnorm_diff_exp_Ysz_T)
                    
                    del_Lx = i/100
                    del_Ysz = 1+k/100
                    del_Ysz_tot =( del_Ysz ** slope_Lx_Ysz) - 1
                    norm_diff_exp_Lx_Ysz = (del_Lx - del_Ysz_tot) * 100
                    Norm_diff_exp_Lx_Ysz.append(norm_diff_exp_Lx_Ysz)
                    errnorm_diff_exp_Lx_Ysz = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Lx_Ysz.append(errnorm_diff_exp_Lx_Ysz)
                    
                    del_R = l/100
                    del_Lx = 1+i/100
                    del_Lx_tot = (del_Lx ** slope_R_Lx) - 1
                    norm_diff_exp_R_Lx = (del_R - del_Lx_tot) * 100
                    Norm_diff_exp_R_Lx.append(norm_diff_exp_R_Lx)
                    errnorm_diff_exp_R_Lx = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_R_Lx.append(errnorm_diff_exp_R_Lx)
                    
                    del_R = l/100
                    del_T = 1+j/100
                    del_T_tot = (del_T ** slope_R_T) - 1
                    norm_diff_exp_R_T = (del_R - del_T_tot) * 100
                    Norm_diff_exp_R_T.append(norm_diff_exp_R_T)
                    errnorm_diff_exp_R_T = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_R_T.append(errnorm_diff_exp_R_T)
                    
                    del_R = l/100
                    del_Ysz = 1+k/100
                    del_Ysz_tot = (del_Ysz ** slope_R_Ysz) - 1
                    norm_diff_exp_R_Ysz = (del_R - del_Ysz_tot) * 100
                    Norm_diff_exp_R_Ysz.append(norm_diff_exp_R_Ysz)
                    errnorm_diff_exp_R_Ysz = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_R_Ysz.append(errnorm_diff_exp_R_Ysz)
                    
                    
                    del_Lx = i/100
                    del_Lbcg = 1+m/100
                    del_Lbcg_tot = (del_Lbcg ** slope_Lx_Lbcg) - 1
                    norm_diff_exp_Lx_Lbcg = (del_Lx - del_Lbcg_tot) * 100
                    Norm_diff_exp_Lx_Lbcg.append(norm_diff_exp_Lx_Lbcg)
                    errnorm_diff_exp_Lx_Lbcg = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Lx_Lbcg.append(errnorm_diff_exp_Lx_Lbcg) 
                    
                    del_Ysz = k/100
                    del_Lbcg = 1+m/100
                    del_Lbcg_tot = (del_Lbcg ** slope_Ysz_Lbcg) - 1
                    norm_diff_exp_Ysz_Lbcg = (del_Ysz - del_Lbcg_tot) * 100
                    Norm_diff_exp_Ysz_Lbcg.append(norm_diff_exp_Ysz_Lbcg)
                    errnorm_diff_exp_Ysz_Lbcg = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Ysz_Lbcg.append(errnorm_diff_exp_Ysz_Lbcg)
                    
                    del_Lbcg = m/100
                    del_T = 1+j/100
                    del_T_tot = (del_T ** slope_Lbcg_T) - 1
                    norm_diff_exp_Lbcg_T = (del_Lbcg - del_T_tot) * 100
                    Norm_diff_exp_Lbcg_T.append(norm_diff_exp_Lbcg_T)
                    errnorm_diff_exp_Lbcg_T = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_Lbcg_T.append(errnorm_diff_exp_Lbcg_T)
                    
                    del_R = l/100
                    del_Lbcg = 1+m/100
                    del_Lbcg_tot = (del_Lbcg ** slope_R_Lbcg) - 1
                    norm_diff_exp_R_Lbcg = (del_R - del_Lbcg_tot) * 100
                    Norm_diff_exp_R_Lbcg.append(norm_diff_exp_R_Lbcg)
                    errnorm_diff_exp_R_Lbcg = np.abs(norm_diff_exp_Lx_T) * np.abs(np.log(del_T) * errslope_Lx_T) * 100
                    errNorm_diff_R_Lbcg.append(errnorm_diff_exp_R_Lbcg)
                    
                    
                    
                    
                    
                    
                    data = np.array([norm_diff_exp_Lx_T, norm_diff_exp_Ysz_T, norm_diff_exp_Lx_Ysz,norm_diff_exp_R_Lx,norm_diff_exp_R_T,norm_diff_exp_R_Ysz,norm_diff_exp_Lx_Lbcg,norm_diff_exp_Ysz_Lbcg,norm_diff_exp_Lbcg_T,norm_diff_exp_R_Lbcg])
                    model = np.array([norm_diff_obs[0],norm_diff_obs[1],norm_diff_obs[2],norm_diff_obs[6],norm_diff_obs[9],norm_diff_obs[7],norm_diff_obs[3],norm_diff_obs[4],norm_diff_obs[5],norm_diff_obs[8]])
                    err = np.array([errnorm_diff_obs[0],errnorm_diff_obs[1],errnorm_diff_obs[2],errnorm_diff_obs[6],errnorm_diff_obs[9],errnorm_diff_obs[7],errnorm_diff_obs[3],errnorm_diff_obs[4],errnorm_diff_obs[5],errnorm_diff_obs[8]])
                    chisq = np.sum(((data-model)/err)**2)
                    chisq_red = chisq/ (len(data)-5) 
                    if chisq_red < 3:
                        chisq_red_all.append(chisq_red)
                        Lx_T_accepted.append(norm_diff_exp_Lx_T)
                        Ysz_T_accepted.append(norm_diff_exp_Ysz_T)
                        Lx_Ysz_accepted.append(norm_diff_exp_Lx_Ysz)
                        Lx_Lbcg_accepted.append(norm_diff_exp_Lx_Lbcg)
                        Ysz_Lbcg_accepted.append(norm_diff_exp_Ysz_Lbcg)
                        Lbcg_T_accepted.append(norm_diff_exp_Lbcg_T)
                        R_Lx_accepted.append(norm_diff_exp_R_Lx)
                        R_T_accepted.append(norm_diff_exp_R_T)
                        R_Ysz_accepted.append(norm_diff_exp_R_Ysz)
                        R_Lbcg_accepted.append(norm_diff_exp_R_Lbcg)

                        err_Lx_T_accepted.append(errnorm_diff_exp_Lx_T)
                        err_Ysz_T_accepted.append(errnorm_diff_exp_Ysz_T)
                        err_Lx_Ysz_accepted.append(errnorm_diff_exp_Lx_Ysz)
                        err_Lx_Lbcg_accepted.append(errnorm_diff_exp_Lx_Lbcg)
                        err_Ysz_Lbcg_accepted.append(errnorm_diff_exp_Ysz_Lbcg)
                        err_Lbcg_T_accepted.append(errnorm_diff_exp_Lbcg_T)
                        err_R_Lx_accepted.append(errnorm_diff_exp_R_Lx)
                        err_R_T_accepted.append(errnorm_diff_exp_R_T)
                        err_R_Ysz_accepted.append(errnorm_diff_exp_R_Ysz)
                        err_R_Lbcg_accepted.append(errnorm_diff_exp_R_Lbcg)

                        I.append(i)
                        J.append(j)
                        K.append(k)
                        L.append(l)
                        M.append(m)

    print(i)
bestfit_bootstrap_dict = {'chisq_red' : chisq_red_all, 'Lx': I, 'Ysz': K, 'T': J, 'R' : L, 'Lbcg' : M, 'Lx_T_accepted' : Lx_T_accepted, 'Ysz_T_accepted':Ysz_T_accepted, 'Lx_Ysz_accepted' : Lx_Ysz_accepted, 'Lx_Lbcg_accepted':Lx_Lbcg_accepted, 'Ysz_Lbcg_accepted' :Ysz_Lbcg_accepted,'Lbcg_T_accepted':Lbcg_T_accepted, 'R_Lx_accepted':R_Lx_accepted, 'R_T_accepted':R_T_accepted, 'R_Ysz_accepted':R_Ysz_accepted, 'R_Lbcg_accepted':R_Lbcg_accepted,'err_Lx_T_accepted' : err_Lx_T_accepted, 'err_Ysz_T_accepted':err_Ysz_T_accepted, 'err_Lx_Ysz_accepted' : err_Lx_Ysz_accepted, 'err_Lx_Lbcg_accepted':err_Lx_Lbcg_accepted, 'err_Ysz_Lbcg_accepted' :err_Ysz_Lbcg_accepted,'err_Lbcg_T_accepted':err_Lbcg_T_accepted, 'err_R_Lx_accepted':err_R_Lx_accepted, 'err_R_T_accepted':err_R_T_accepted, 'err_R_Ysz_accepted':err_R_Ysz_accepted, 'err_R_Lbcg_accepted':err_R_Lbcg_accepted}              
bestfit_bootstrap = pd.DataFrame(bestfit_bootstrap_dict) 
bestfit_bootstrap.to_csv('percent_diff_accepted_ccVncc_Clusters(step_5)_test_1.csv')

bins=10
plt.hist(I,bins=bins)
plt.xlabel('del_Lx')
plt.show()

plt.hist(J,bins=bins)
plt.xlabel('del_T')
plt.show()

plt.hist(K,bins=bins)
plt.xlabel('del_Ysz')
plt.show()

plt.hist(L,bins=3)
plt.xlabel('del_R')
plt.show()

plt.hist(M,bins=bins)
plt.xlabel('del_Lbcg')
plt.show()

print(np.min(I), np.max(I))
print(np.min(J), np.max(J))
print(np.min(K), np.max(K))
print(np.min(L), np.max(L))
print(np.min(M), np.max(M))