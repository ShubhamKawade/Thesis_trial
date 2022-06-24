#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:26:54 2021

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 






#Verifying the old and the new M cut
# =============================================================================
# groups = ['A0168','A0189','A0262','A0400','A0407','A0957','A1060','A1142','A1185','A1991','A2064','A2151a','A2634','A2717','A2877','A3341','A3392','A3526+','A3558C','A3581','A3744','AWM4','AWM5','CAN010','CID28','HCG62','IC1262','IVZw038','MKW11','MKW4','MKW8','NGC1132','NGC1550','NGC4325','NGC4636','NGC5044+','NGC5846','NGC6338i','RBS0540','RXCJ1252.5-3116','RXCJ1304.2-3030','RXJ0123.2+3327','RXJ0123.6+3315','RXJ0228.2+2811','RXJ0341.3+1524','RXJ1205.1+3920','S0301','S0540','S0753','S0805','S0851','UGC03957','USGCS152','Zw0959.6+3257','ZwCl1665','ZwCl8338','A1035','A1648','A2399','NGC7556','PegasusII','RXCJ0340.6m0239','S0384','S0868','A2622','A3570','A3733','RXCJ1353.4m2753','RXJ1740.5p3539','RXCJ2104.9m5149','S0987','S0555','S1136','Zw1420.2p4952','NGC1650','CID36','RXCJ1337.4m4120','RXCJ1742.8p3900','A0194','RXCJ2124.3m7446','RXCJ1926.9m5342','S0112','A0076']
# mass = pd.read_csv('/home/schubham/Thesis/Thesis/Data/eeHIF_masses.csv')
# groups_eehif = mass[mass.M_weighted < 1]
# groups_eehif = groups_eehif.reset_index(drop=True)
# 
# 
# groups = pd.Series(groups)
# groups.columns = ['Cluster']
# I = []
# for i in range(len(groups)):
#     if groups_eehif['Cluster'][i].casefold() == groups[:][i].casefold():
#         I.append(i)
#         groups[:].str.casefold()
# len(I)
# 
# 
# 
# =============================================================================


omega_m = 0.3
omega_lambda = 0.7
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

### Calculating masses with the MCXC file from Frederic

# =============================================================================
# data = pd.read_csv('/home/schubham/Thesis/Thesis/Data/mcxc_full_data_YszUpd_Shubham.csv')
# 
# Z = (data['Redshift z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# D_a = cosmo.angular_diameter_distance(Z) * 1000
# 
# Lx = data['Lx [10^44 erg s^-1]']
# sigma_Lx =  data['sig_Lx [10^44 erg s^-1]']
# Lx_new = Lx/E
# 
# 
# a = 1.15   #(slope)
# sigma_a = 0.395
# exp_p = 4.70  # ycept
# sigma_exp_p = 1.465
# C_M = 7.41e14
# scatter = 0.54 *0.4343 # to convert to log space
# 
# M_halo_Lx = C_M * (Lx_new**(1/a)) * (1/exp_p**(1/a)) * (1/E) 
# 
# 
# 
# S = Lx_new
# sigma_S = sigma_Lx
# M_halo = M_halo_Lx
# sig_a = M_halo * (1/S) * (1/a) * sigma_S
# sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
# sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p
# sigma_M_Lx = np.sqrt( sig_a**2 + sig_b**2 + sig_c**2 )
# 
# 
# # From Ysz-M
# Ysz = data['Ysz [kpc^2]']
# sigma_Ysz = data['sig_Ysz [kpc^2]']
# Z = (data['Redshift z'])
# E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
# Ysz_new = Ysz*E/10
# 
# a = 1.14   #(slope)
# sigma_a = 0.155
# exp_p = 11.10  # ycept
# sigma_exp_p = 0.925
# scatter = 0.29 * 0.4343 # to convert to the log space
# C_M = 7.41e14 #M_sun
# S = Ysz_new
# sigma_S = sigma_Ysz*E/10
# 
# for i in range(len(Ysz)):
#     
#     
#     if Ysz[i]> 0:
#         M_halo_Ysz = C_M * Ysz_new**(1/a) * (1/exp_p)**(1/a) * (1/E)
#         weights_Ysz = np.ones_like(M_halo_Ysz)/len(M_halo_Ysz)
#     
#     # To calculate sig_M from Mulroy 2019
#         M_halo = M_halo_Ysz
#         sig_a = M_halo * (1/S) * (1/a) * sigma_S
#         sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
#         sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p
#         sigma_M_Ysz = np.sqrt( sig_a**2 + sig_b**2 + sig_c**2 )
#     elif Ysz[i]<0 :
#         M_halo_Ysz = 0
#         sigma_M_Ysz = 0
# data[data['Cluster'] == 'A0189']
# M_halo_Ysz[125]
# =============================================================================



# T-M from master_file_new
file = pd.read_csv('/users/kshubham/Documents/Shubham_thesis/Data/R_total.csv')
file = general_functions.cleanup(file)
file = file.fillna(0)
file = file.reset_index(drop=True)
Z = (file['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000
T_chandra = file['T']
T_max_chandra = file['Tmax']
T_min_chandra = file['Tmin']
T = T_chandra **1.125 # xmm temperature
T_plus = T_max_chandra - T_chandra
T_minus = T_chandra - T_min_chandra
sigma_T_chandra = (T_plus+T_minus)/2
sigma_T = (T_chandra**0.125) * sigma_T_chandra * 1.125

a = 0.47   #(slope)
sigma_a = 0.105
exp_p = 6.98  # ycept
sigma_exp_p = 0.445
scatter = 0.2# To convert to the log space
## Scatter needs to be added to Norm uncertities 
sigma_exp_p_wscatter = np.sqrt((sigma_exp_p/exp_p)**2 + scatter**2) * exp_p
C_M = 7.41e14 #M_sun 

M_halo_T = (C_M/E) * T**(1/a) * 1/(exp_p**(1/a))
weights_T = np.ones_like(M_halo_T)/len(M_halo_T)

S = T
sigma_S = sigma_T
M_halo = M_halo_T
sigma_T_chandra[90]

sig_a = M_halo * (1/S) * (1/a) * sigma_S
sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p_wscatter
sig_scatter = 0.2 * 0.4343
sigma_M_T = np.sqrt( sig_a**2 + sig_b**2 + sig_c**2 )

########### From Lx-M ############################

Z = (file['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
Lx = file['Lx']
err_Y = file['eL']
Lx_new = Lx/E
sigma_Lx = err_Y*Lx_new/100



a = 1.15  #(slope)
sigma_a = 0.395
exp_p = 4.70 # ycept
sigma_exp_p = 1.465
scatter = 0.54# To convert to the log space
## Scatter needs to be added to Norm uncertities 
sigma_exp_p_wscatter = np.sqrt((sigma_exp_p/exp_p)**2 + scatter**2) * exp_p

C_M = 7.41e14 #M_sun 

M_halo_Lx = (C_M/E) * Lx_new**(1/a) * 1/(exp_p**(1/a))
weights_Lx = np.ones_like(M_halo_Lx)/len(M_halo_Lx)

S = Lx_new
sigma_S = sigma_Lx
M_halo = M_halo_Lx

sig_a = M_halo * (1/S) * (1/a) * sigma_S
sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p_wscatter
sig_scatter = 0.2 * 0.4343
sigma_M_Lx = np.sqrt( sig_a**2 + sig_b**2 + sig_c**2 )

# =============================================================================
# file_mcxc = pd.read_fwf('/users/kshubham/Documents/Shubham_thesis/Data/MCXC-Ysz-Jens-FINAL.txt')
# file_mcxc.to_csv('MCXC_full.csv', sep = ',')
# =============================================================================
########### From Ysz-M  #########################
### This is to get M_halo for clusters with Ysz>0
mcxc = pd.read_csv('/users/kshubham/Documents/Shubham_thesis/Data/MCXC_full.csv')
Ysz_mcxc_arcmin = mcxc['Ysz']
Z_mcxc = mcxc['z']
E_mcxc = (omega_m*(1+Z_mcxc)**3 + omega_lambda)**0.5
D_a_mcxc = cosmo.angular_diameter_distance(Z_mcxc) * 1000
e_Y_arcmin = mcxc['eY']
Ysz_mcxc_new = (Ysz_mcxc_arcmin * (D_a_mcxc.value**2) * (np.pi / (60*180))**2)  * E_mcxc / 20
sigma_Ysz_mcxc = (e_Y_arcmin * (D_a_mcxc.value**2) * (np.pi / (60*180))**2) * E_mcxc/20

#eehif_pos_Ysz = eehif[eehif['Y(r/no_ksz,arcmin^2)'] > 0]
Z = (file['z'])
E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
D_a = cosmo.angular_diameter_distance(Z) * 1000
file[file['Cluster'] == 'A1142']
Ysz_arcmin = file['Ysz']
e_Y_arcmin = file['eY']
Ysz = (Ysz_arcmin * (D_a.value**2) * (np.pi / (60*180))**2)
np.median(Ysz)

Ysz_new = Ysz*E/20
sigma_Ysz = ((e_Y_arcmin)* (D_a.value**2) * (np.pi / (60*180))**2)*E/20   

a = 1.14  #(slope)
sigma_a = 0.155
exp_p = 11.10 # ycept
sigma_exp_p = 0.925
scatter = 0.29 #in ln space
## Scatter needs to be added to Norm uncertities 
sigma_exp_p_wscatter = np.sqrt((sigma_exp_p/exp_p)**2 + scatter**2) * exp_p
C_M = 7.41e14 #M_sun 

# =============================================================================
# for i in range(len(file)):
# 
#     if Ysz[i] < 0:
#         
#         p = file['MCXC'][i]
#         print(p)
#         index = mcxc.index[mcxc['Cluster']==p]
#         print(index)
# 
# ========================================================================= = []+

M_halo_Ysz = []
for i in range(len(file)):
# =============================================================================
#     if i == 245 :
#         m_halo_Ysz = 0.0
#         M_halo_Ysz.append(M_halo_Ysz)
# =============================================================================
    if Ysz[i] < 0:
        
        p = file['MCXC'][i]
        index = mcxc.index[mcxc['Cluster']==p][0]
        if Ysz_mcxc_new[index] < 0:
             m_halo_Ysz = 0.0
             M_halo_Ysz.append(m_halo_Ysz)
        else :
            m_halo_Ysz = (C_M/E_mcxc[index]) * Ysz_mcxc_new[index]**(1/a) * 1/(exp_p**(1/a))
            M_halo_Ysz.append(m_halo_Ysz)
            #weights_Ysz = np.ones_like(M_halo_Ysz)/len(M_halo_Ysz)
    else:
        m_halo_Ysz = (C_M/E[i]) * Ysz_new[i]**(1/a) * 1/(exp_p**(1/a))
        M_halo_Ysz.append(m_halo_Ysz)
        

#file[file['M_Ysz_test'].isnull()]
# =============================================================================
# S = Ysz_new
# sigma_S = sigma_Ysz
# M_halo = M_halo_Ysz
# 
# sig_a = M_halo * (1/S) * (1/a) * sigma_S
# sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
# sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p_wscatter
# sigma_M_Ysz = np.sqrt( sig_a**2 + sig_b**2 + sig_c**2 )
# =============================================================================

sigma_M_Ysz = []
for i in range(len(file)):
# =============================================================================
#     if i == 245 :
#         sigma_m_Ysz = 0.0
#         sigma_M_Ysz.append(sigma_m_Ysz)
# =============================================================================
    if Ysz[i] < 0:
        
        p = file['MCXC'][i]
        index = mcxc.index[mcxc['Cluster']==p][0]
        if Ysz_mcxc_new[index] < 0:
             sigma_m_Ysz = 0.0
             sigma_M_Ysz.append(sigma_m_Ysz)
        else :
            S = Ysz_mcxc_new[index]
            sigma_S = sigma_Ysz_mcxc[index]
            M_halo = M_halo_Ysz[i]
            
            sig_a = M_halo * (1/S) * (1/a) * sigma_S
            sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
            sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p_wscatter
            sigma_m_Ysz = ( sig_a**2 + sig_b**2 + sig_c**2 )**0.5
            sigma_M_Ysz.append(sigma_m_Ysz)

            #weights_Ysz = np.ones_like(M_halo_Ysz)/len(M_halo_Ysz)
    else:
        S = Ysz_new[i]
        sigma_S = sigma_Ysz[i]
        M_halo = M_halo_Ysz[i]
        
        sig_a = M_halo * (1/S) * (1/a) * sigma_S
        sig_b = M_halo * (np.log(exp_p) - np.log(S) ) * (1/a**2) * sigma_a
        sig_c = M_halo * (-1/a) * (1/exp_p) * sigma_exp_p_wscatter
        sigma_m_Ysz = ( sig_a**2 + sig_b**2 + sig_c**2 )**0.5
        sigma_M_Ysz.append(sigma_m_Ysz)


sigma_M_Ysz[27]
#Calculate the weighted mean of the three masses
#add the masses to the 
file['M_halo_Lx'] = M_halo_Lx
file['M_halo_Ysz'] = M_halo_Ysz
file['M_halo_T'] = M_halo_T
file['err_M_Lx'] = sigma_M_Lx
file['err_M_Ysz'] = sigma_M_Ysz
file['err_M_T'] = sigma_M_T
file.fillna(0,inplace=True)



M_halo_Ysz = file['M_halo_Ysz']/1e14 
M_halo_Lx = file['M_halo_Lx'] /1e14
M_halo_T = file['M_halo_T']/1e14
sigma_M_Lx = file['err_M_Lx']/1e14
sigma_M_Ysz= file['err_M_Ysz'] /1e14 
sigma_M_T = file['err_M_T']  /1e14

M_weighted = []
sigma_M_weighted = []

for i in range(len(file)):
    
    
    if (M_halo_T[i] == 0) and (M_halo_Ysz[i] == 0): 
        m_new = 1e14
        sigma_m_tot = 0
        M_weighted.append(m_new)
        sigma_M_weighted.append(sigma_m_tot) 
        
        
    if (M_halo_T[i] == 0) and (M_halo_Ysz[i] != 0):
        
        weights = ((1/sigma_M_Lx[i]**2)  + (1/sigma_M_Ysz[i]**2))
        m = (M_halo_Lx[i]/(sigma_M_Lx[i])**2)  + (M_halo_Ysz[i]/(sigma_M_Ysz[i])**2)
        mtot = m/weights
        
        m_new = (M_halo_Lx[i]**2/(sigma_M_Lx[i])**2)  + (M_halo_Ysz[i]**2/(sigma_M_Ysz[i])**2)
        sigma_m_tot = np.sqrt(1/(2-1) * ((m_new/weights) - mtot**2))
        M_weighted.append(m_new)
        sigma_M_weighted.append(sigma_m_tot)
        
    if (M_halo_Ysz[i] == 0) and (M_halo_T[i] != 0) :
        weights = ((1/sigma_M_Lx[i]**2)  + (1/sigma_M_T[i]**2))
        m = (M_halo_Lx[i]/(sigma_M_Lx[i])**2)  + (M_halo_T[i]/(sigma_M_T[i])**2)
        mtot = m/weights
        
        m_new = (M_halo_Lx[i]**2/(sigma_M_Lx[i])**2)  + (M_halo_T[i]**2/(sigma_M_T[i])**2)
        sigma_m_tot = np.sqrt(1/(2-1) * ((m_new/weights) - mtot**2))
        M_weighted.append(m_new)
        sigma_M_weighted.append(sigma_m_tot)
    
    
    if (M_halo_Ysz[i] != 0) and (M_halo_T[i] != 0) and (M_halo_Lx[i] != 0) :
        
        weights = ((1/sigma_M_Lx[i]**2) + (1/sigma_M_Ysz[i]**2) + (1/sigma_M_T[i]**2))
        m = (M_halo_Lx[i]/(sigma_M_Lx[i])**2) + (M_halo_Ysz[i]/(sigma_M_Ysz[i])**2) + (M_halo_T[i]/(sigma_M_T[i])**2)
        mtot = m/weights
        
        m_new = (M_halo_Lx[i]**2/(sigma_M_Lx[i])**2) + (M_halo_Ysz[i]**2/(sigma_M_Ysz[i])**2) + (M_halo_T[i]**2/(sigma_M_T[i])**2)
        sigma_m_tot = np.sqrt(1/(3-1) * ((m_new/weights) - mtot**2))
        M_weighted.append(m_new)
        sigma_M_weighted.append(sigma_m_tot)
    print(i)
## file dataframe was used to combine all te M estimates and determine M_weighted
# and then it was added to master_file and renamed as file
file['M_weighted'] = M_weighted
file['sigma_M_weighted'] = sigma_M_weighted
file.to_csv('mass_estimates_all.csv')



groups = file[file.M_weighted < 1]
groups
file.loc[69]
master_file = pd.read_csv('/users/kshubham/Documents/Shubham_thesis/Data/master_file_mass.csv')
master_file[master_file.Cluster == 'A3158']
master_file.loc[131]

M_weighted[304]
eehif['M_weighted'][304]

for i in range(len(eehif)):
    
    if M_weighted[i] != eehif['M_weighted'][i]:
        print(eehif.index[eehif['M_weighted'] == eehif['M_weighted'][i]])






#eehif['M_weighted'] = M_weighted
groups = eehif[eehif['M_weighted'] < 1e14]
list_groups = groups['Cluster']
list_groups.to_csv('groups_eeHIF.csv')
group_names = []
for i in range(len(list_groups)):
    group_names.append(list_groups.iloc[i])

def removing_galaxy_groups(data):
    groups = ['A0168',
 'A0189',
 'A0262',
 'A0400',
 'A0407',
 'A0957',
 'A1060',
 'A1142',
 'A1185',
 'A1991',
 'A2064',
 'A2151a',
 'A2634',
 'A2717',
 'A2877',
 'A3341',
 'A3392',
 'A3526+',
 'A3558C',
 'A3581',
 'A3744',
 'AWM4',
 'AWM5',
 'CAN010',
 'CID28',
 'HCG62',
 'IC1262',
 'IVZw038',
 'MKW11',
 'MKW4',
 'MKW8',
 'NGC1132',
 'NGC1550',
 'NGC4325',
 'NGC4636',
 'NGC5044+',
 'NGC5846',
 'NGC6338i',
 'RBS0540',
 'RXCJ1252.5-3116',
 'RXCJ1304.2-3030',
 'RXJ0123.2+3327',
 'RXJ0123.6+3315',
 'RXJ0228.2+2811',
 'RXJ0341.3+1524',
 'RXJ1205.1+3920',
 'S0301',
 'S0540',
 'S0753',
 'S0805',
 'S0851',
 'UGC03957',
 'USGCS152',
 'Zw0959.6+3257',
 'ZwCl1665',
 'ZwCl8338',
 'A1035',
 'A1648',
 'A2399',
 'NGC7556',
 'PegasusII',
 'RXCJ0340.6m0239',
 'S0384',
 'S0868',
 'A2622',
 'A3570',
 'A3733',
 'RXCJ1353.4m2753',
 'RXJ1740.5p3539',
 'RXCJ2104.9m5149',
 'S0987',
 'S0555',
 'S1136',
 'Zw1420.2p4952',
 'NGC1650',
 'CID36',
 'RXCJ1337.4m4120',
 'RXCJ1742.8p3900',
 'A0194',
 'RXCJ2124.3m7446',
 'RXCJ1926.9m5342',
 'S0112',
 'A0076']
    for i in range(len(groups)):
        data = data.drop(data[data['Cluster']==f'{groups[i]}'].index, inplace=True)
    return data




for i in range(len(groups)):
    sz.drop(sz[sz['Cluster']==f'{groups[i]}'].index, inplace=True)

##########################################################################################
                                       # Plotting histogram
bins_Lx = np.arange(0, np.max(M_halo_Lx)+0.1e15, 0.1e15)
bins_Ysz = np.arange(0, np.max(M_halo_Ysz)+0.1e15, 0.1e15)
bins_T = np.arange(0, np.max(M_halo_T)+0.1e15, 0.1e15)

plt.hist(M_halo_Lx, bins=bins_Lx, weights = weights_Lx, color = 'green', alpha = 1, histtype = 'step', label = 'Lx-M')
plt.hist(M_halo_Ysz, bins= bins_Ysz, weights = weights_Ysz, color = 'blue', alpha = 1,histtype = 'step',label = 'Ysz-M')
plt.hist(M_halo_T, bins= bins_T, weights = weights_T, color = 'red', alpha = 1,histtype = 'step',label = 'T-M')
plt.xlim(right = 2e15)
#plt.xlim(0,20)
plt.xlabel('$M_{halo}$ [$M_{\odot}$]')
plt.ylabel('Count')
plt.title('Cluster masses based M')
plt.legend()
plt.show()


####### Plotting comparison scatter plots
sz = pd.read_csv('/home/schubham/Thesis/Thesis/Data/master_file_mass.csv')
M_halo_Ysz = sz['M_halo_Ysz']/1e15 
M_halo_Lx = sz['M_halo_Lx']/1e15 
M_halo_T = sz['M_halo_T']/1e15 

bins_Lx = np.arange(0, np.max(M_halo_Lx)+0.3, 0.3)
bins_Ysz = np.arange(0, np.max(M_halo_Ysz)+0.3, 0.3)
bins_T = np.arange(0, np.max(M_halo_T)+0.3, 0.3)

# =============================================================================
# bins_Lx = 10
# bins_Ysz = 10
# bins_T = 10
# =============================================================================

weights_Lx = np.ones_like(M_halo_Lx)/ len(M_halo_Lx)
weights_Ysz = np.ones_like(M_halo_Ysz)/ len(M_halo_Ysz)
weights_T = np.ones_like(M_halo_T)/ len(M_halo_T)

plt.hist(M_halo_Lx, bins=bins_Lx, weights = weights_Lx, color = 'green', alpha = 1, histtype = 'step', label = r'$L_{\mathrm{X}}-M$')
plt.hist(M_halo_Ysz, bins= bins_Ysz, weights = weights_Ysz, color = 'blue', alpha = 1,histtype = 'step',label = r'$Y_{\mathrm{SZ}}-M$')
plt.hist(M_halo_T, bins= bins_T, weights = weights_T, color = 'red', alpha = 1,histtype = 'step',label = '$T-M$')
#plt.xlim(right = 2e15)
#plt.xlim(0,20)
plt.xlabel(r'$M_{\mathrm{halo}}$ (*$10^{14}\,\mathrm{M}_{\odot}$)')
plt.ylabel('Count')
plt.xscale('log')
#plt.xlim(0.2,15)
plt.title('Cluster mass')
plt.legend()
plt.savefig('/home/schubham/Thesis/Thesis/Plots/M_histogram.png',dpi=300,bbox_inches="tight")
plt.show()

# For M_L vs M_T
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(np.log10(M_L),np.zeros_like(M_L),np.log10(M_T),np.zeros_like(M_T))

z = Norm * M_L**Slope
plt.scatter(M_L, M_T, s = 15)
plt.plot(M_L, z, color = 'red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$M_{L_{X}-M}$ [$M_{\odot}$]')
plt.ylabel('$M_{T-M}$ [$M_{\odot}]$')
plt.title('Mass comparison from $L_{X} - M$ and T - M')
plt.show()

## For M_L vs M_Ysz
ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(np.log10(M_L),np.zeros_like(M_L),np.log10(M_Y),np.zeros_like(M_Y))

z = Norm * M_L**Slope
plt.errorbar(M_L, M_Y,sigma_M_Lx,sigma_M_T)
#plt.plot(M_L, z, color = 'red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$M_{L_{X}-M}$ [$M_{\odot}$]')
plt.ylabel('$M_{Y_{sz}-M}$ [$M_{\odot}]$')
plt.title('Mass comparison from $L_{X} - M$ and $Y_{SZ}$ - M')
plt.show()


ycept,Norm,Slope,Scatter = general_functions.calculate_bestfit(np.log10(M_Y),np.zeros_like(M_Y),np.log10(M_T),np.zeros_like(M_T))

z = Norm * M_Y**Slope
plt.scatter(M_Y, M_T, s = 15)
plt.plot(M_Y, z, color = 'red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$M_{Y_{SZ}-M}$ [$M_{\odot}$]')
plt.ylabel('$M_{T-M}$ [$M_{\odot}]$')
plt.title('Mass comparison from $Y_{SZ} - M$ and T - M')
plt.show()
np.log(6.98)
sz[sz['Cluster'] == 'A3558C']
## Uncertainty propagation
a = 1.15
sigma_a = 0.395
p = 1.943
sigma_p = 0.063
A = T**(1/a)
B = (np.exp(p))**(-1/a)
sigma_A = A*np.sqrt( (((1/a)/T)*sigma_T) ** 2 + (np.log(T)*sigma_a/(a**2))**2)
sigma_B = np.sqrt( (np.exp(p)) **(-2/p) * (a**2 * (sigma_p)**2 + (sigma_a)**2 * (np.log(np.exp(p)))**2 ))/a**2  

sigma_M = (C_M/E) * np.sqrt((sigma_A/A)**2 + (sigma_B/B)**2)                              

