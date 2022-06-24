#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:28:22 2022

@author: schubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions
from astropy.cosmology import LambdaCDM 

for j in range(1,1001,1):
    print(j)
    
    sample = pd.read_csv(f'/home/schubham/Thesis/Thesis/Simulated_clusters/simul_samples/sample_{j}.txt', sep = '\\s+')
    sample = sample.set_axis(['Cluster', 'z', 'Glon','Glat', 'T', 'Tmax', 'Tmin', 'Lx', 'eL', 'Y', 'eY', 'scatter_Lx', 'scatter_Ysz'], axis=1)
    sample = general_functions.cleanup(sample)
    
    sample['Y'] = sample['Y'] *(np.pi/10800)**2 * (1000**2)
    sample['eY'] = sample['eY'] *(np.pi/10800)**2 * (1000**2)
    sample.rename({'Y':'Y_kpc^2','eY':'eY_kpc^2'},axis=1,inplace=True)
    
    #sample.to_csv('sample_1.csv', sep = ',')

    omega_m = 0.3
    omega_lambda = 0.7
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    
    
    
    # T-M from master_file_new
   
    Z = (sample['z'])
    E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
    D_a = cosmo.angular_diameter_distance(Z) * 1000
    T_chandra = sample['T']
    T_max_chandra = sample['Tmax']
    T_min_chandra = sample['Tmin']
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
    
    Z = (sample['z'])
    E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
    Lx = sample['Lx']
    err_Y = sample['eL']
    Lx_new = Lx
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
    # sample_mcxc = pd.read_fwf('/users/kshubham/Documents/Shubham_thesis/Data/MCXC-Ysz-Jens-FINAL.txt')
    # sample_mcxc.to_csv('MCXC_full.csv', sep = ',')
    # =============================================================================
    ########### From Ysz-M  #########################
    ### This is to get M_halo for clusters with Ysz>0
    mcxc = pd.read_csv('/home/schubham/Thesis/Thesis/Data/MCXC_full.csv')
    
    Ysz_mcxc_arcmin = mcxc['Ysz']
    Z_mcxc = mcxc['z']
    E_mcxc = (omega_m*(1+Z_mcxc)**3 + omega_lambda)**0.5
    D_a_mcxc = cosmo.angular_diameter_distance(Z_mcxc) * 1000
    e_Y_arcmin = mcxc['eYsz']
    Ysz_mcxc_new = (Ysz_mcxc_arcmin * (D_a_mcxc.value**2) * (np.pi / (60*180))**2)  * E_mcxc / 20
    sigma_Ysz_mcxc = (e_Y_arcmin * (D_a_mcxc.value**2) * (np.pi / (60*180))**2) * E_mcxc/20
    
    #eehif_pos_Ysz = eehif[eehif['Y(r/no_ksz,arcmin^2)'] > 0]
    Z = (sample['z'])
    E = (omega_m*(1+Z)**3 + omega_lambda)**0.5
    D_a = cosmo.angular_diameter_distance(Z) * 1000
    sample[sample['Cluster'] == 'A1142']
    Ysz = sample['Y_kpc^2']
    Ysz_new = sample['Y_kpc^2']/20
    sigma_Ysz = sample['eY_kpc^2']/20
   
   
    
    a = 1.14  #(slope)
    sigma_a = 0.155
    exp_p = 11.10 # ycept
    sigma_exp_p = 0.925
    scatter = 0.29 #in ln space
    ## Scatter needs to be added to Norm uncertities 
    sigma_exp_p_wscatter = np.sqrt((sigma_exp_p/exp_p)**2 + scatter**2) * exp_p
    C_M = 7.41e14 #M_sun 
    
    # =============================================================================
    # for i in range(len(sample)):
    # 
    #     if Ysz[i] < 0:
    #         
    #         p = sample['MCXC'][i]
    #         print(p)
    #         index = mcxc.index[mcxc['Cluster']==p]
    #         print(index)
    # 
    # ========================================================================= = []+
    
    M_halo_Ysz = []
    for i in range(len(sample)):
    # =============================================================================
    #     if i == 245 :
    #         m_halo_Ysz = 0.0
    #         M_halo_Ysz.append(M_halo_Ysz)
    # =============================================================================
        if Ysz[i] < 0:
            
            p = sample['MCXC'][i]
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
            
    
    #sample[sample['M_Ysz_test'].isnull()]
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
    for i in range(len(sample)):
    # =============================================================================
    #     if i == 245 :
    #         sigma_m_Ysz = 0.0
    #         sigma_M_Ysz.append(sigma_m_Ysz)
    # =============================================================================
        if Ysz[i] < 0:
            
            p = sample['MCXC'][i]
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
    sample['M_halo_Lx'] = M_halo_Lx
    sample['M_halo_Ysz'] = M_halo_Ysz
    sample['M_halo_T'] = M_halo_T
    sample['err_M_Lx'] = sigma_M_Lx
    sample['err_M_Ysz'] = sigma_M_Ysz
    sample['err_M_T'] = sigma_M_T
    sample.fillna(0,inplace=True)
    
    
    
    M_halo_Ysz = sample['M_halo_Ysz']/1e14 
    M_halo_Lx = sample['M_halo_Lx'] /1e14
    M_halo_T = sample['M_halo_T']/1e14
    sigma_M_Lx = sample['err_M_Lx']/1e14
    sigma_M_Ysz= sample['err_M_Ysz'] /1e14 
    sigma_M_T = sample['err_M_T']  /1e14
    
    M_weighted = []
    sigma_M_weighted = []
    
    for i in range(len(sample)):
        
        
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
        
    ## sample dataframe was used to combine all te M estimates and determine M_weighted
    # and then it was added to master_sample and renamed as sample
    sample['M_weighted_1e14'] = M_weighted
    sample['sigma_M_weighted_1e14'] = sigma_M_weighted
    sample.to_csv(f'/home/schubham/Thesis/Thesis/Simulated_clusters/simulated_clusters_with_mass/sample_{j}.csv')