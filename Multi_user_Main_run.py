#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:41:51 2023

@author: dddd
"""

import numpy as np
import math as ma
import time
import timeit
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv

import Multi_user_ENV as MultiENV
import function_compute as FUNC
import Genetic_algorithm as GA_init
import SimpleGA as sga
import Exhaustive_search as Exh
import WOA as woa
import GWOA as gwoa
import WOGA as woga
import DWOA as dwoa
import EE_AO_SCA as sca

from Multi_user_BB import MultiU_BB
import EE_RIS_interference as EE_RIS
import EE_SR_BD as SRBD

from SE_Multi_user_BB import MultiU_BB_SE
import SE_RIS_interference as SE_RIS
import SE_SR_BD as SEBD
import RIS_SR_SU as srsu



seed = 15
np.random.seed(seed)

NumPUE = 4
NumSUE = 4
NumUE = NumPUE + NumSUE

RIS_bits = 3
RIS_Lnum = 1
Num_RISele = 16
GroupRIS = 2
NumBSAnt = 8
L_spreadingfactor = 10
P_Rmin = 2.5
S_Rmin = 1

P_Rmin_RIS = 0.1
S_Rmin_RIS = 0.1

Bandwidth = 20000000
# sigma2 = 10**(-104/10) * 10**3
# Bandwidth=1 #MHz
Noise_o=-104 #dBm  -174dBm/Hz
sigma2=pow(10, (Noise_o/10))/pow(10,3) #watts


#power limits

mu = 1.2                                   #---------power amplifier efficiency ^ -1---------#

P_max_o=45 #dBm
P_max=pow(10, (P_max_o/10))/pow(10,3)    #------------Maximum transmitted power------------#
P_k_o=10  #dBm
P_k=pow(10, (P_k_o/10))/pow(10,3)        #----------User static power comsumption----------#

PT_static = 39 #dbm
PT_static = pow(10, (PT_static/10))/pow(10,3)
RIS_static = 10 #dbm
RIS_static = pow(10, (RIS_static/10))/pow(10,3)
PUE_static = 10 #dbm
PUE_static = pow(10, (PUE_static/10))/pow(10,3)
SUE_static = 10 #dbm
SUE_static = pow(10, (SUE_static/10))/pow(10,3)
SIC_dissapation = 0.2 #W



# #------------BS location-----------------#

BS_loc=np.array([-250,-250, 10])



# #-----------RIS location-----------------#

RISloc=np.zeros([RIS_Lnum, 3])
for i in range(0,RIS_Lnum):   # i = l_RIS, i從0開始

    RISloc[i] = [0, 0, 20]
    
    

#---------------User location-----------------#

User_loc = np.zeros([NumUE, 3])

#--------------Ramdom or Fixed user location------------#
UE_count = 0
while(UE_count < NumPUE):
    User_loc[UE_count][0] = 30*np.random.rand(1)
    User_loc[UE_count][1] = 30*np.random.rand(1)
    User_loc[UE_count][2] = 1
        
    UE_count += 1
    

while(UE_count < NumUE):

    User_loc[UE_count] = User_loc[UE_count - NumPUE]
    User_loc[UE_count][0] += 1 
    
    UE_count += 1

# print("UE 0: ",  User_loc[0], ",\nUE 1 : ",  User_loc[1], ",\nUE 2 : ", User_loc[2], ",\nUE 3 : ", User_loc[3])
        
dismin = 10
PL_0 = ma.pow(10, (-30/10))

MultiUE = MultiENV.MultiUE_ENV(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE)
Funcom = FUNC.opt_function(NumBSAnt, Num_RISele, NumUE, sigma2)

#===================================================================================================================#

P_pathloss_B2U, pathloss_B2R, P_pathloss_R2U, S_pathloss_B2U, S_pathloss_R2U = MultiUE.H_GenPL(BS_loc, RISloc[0], User_loc)
# print("P_B2U : ", P_pathloss_B2U, ",\nB2R : ",  pathloss_B2R, ",\nP_R2U : ", P_pathloss_R2U, "\nS_B2U : ", S_pathloss_B2U, "\nS_R2U : ", S_pathloss_R2U)

P_B2U_NLoS, B2R_NLoS, P_R2U_NLoS, S_B2U_NLoS, S_R2U_NLoS = MultiUE.H_GenNLoS()
# print("P_B2U : ", P_B2U_NLoS.shape, ",\nB2R : ",  B2R_NLoS.shape, ",\nP_R2U : ", P_R2U_NLoS.shape, ",\nS_B2U : ", S_B2U_NLoS.shape, ",\nS_R2U : ", S_R2U_NLoS.shape)

P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray = MultiUE.H_RayleighOverall(P_B2U_NLoS, B2R_NLoS, P_R2U_NLoS, S_B2U_NLoS, 
                S_R2U_NLoS, P_pathloss_B2U, pathloss_B2R, P_pathloss_R2U, S_pathloss_B2U, S_pathloss_R2U)
# print("P_B2U : ", P_B2U_Ray.shape, ",\nB2R : ",   B2R_Ray.shape, ",\nP_R2U : ", P_R2U_Ray.shape, "\nS_B2U : ", S_B2U_Ray.shape, ",\nS_R2U : ", S_R2U_Ray.shape)

# =================================================================================================================== #
# --------------------------------------------------Number SU compare------------------------------------------------ #
# =================================================================================================================== #

# SU_times = int(NumSUE/NumPUE)
# # SU_times = ma.ceil(NumSUE/NumPUE)
# SRSU_power_ini = 21
# SRSU_theta_ini = 0
# SRSU_onoff_ini = 1

# SRSU_start = timeit.default_timer()

# SRSU_status, SRSU_objective, power_SRSU, theta_SRSU, Time_SRSU = srsu.SR_SU(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2, L_spreadingfactor,
#           P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, SRSU_power_ini, SRSU_theta_ini, SRSU_onoff_ini, P_Rmin, S_Rmin, SU_times)

# SRSU_end = timeit.default_timer()
# Time_SRSU += (SRSU_end - SRSU_start)
# print("BB time use : ", Time_SRSU)

# =================================================================================================================== #
# --------------------------------------------------------EE AO SCA-------------------------------------------------- #
# =================================================================================================================== #

# SCA_SR_power_ini = 30
# SCA_SR_theta_ini = 0
# SCA_SR_onoff_ini = 1

# SCA_status = sca.EE_SCA(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           L_spreadingfactor, P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, SCA_SR_power_ini,  SCA_SR_theta_ini,
#           SCA_SR_onoff_ini, P_Rmin, S_Rmin)

# SCA_status, SCA_objective, EE_power_SCA, EE_theta_SCA, EE_SEBD_SCA = sca.EE_SCA(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           L_spreadingfactor, P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, SCA_SR_power_ini,  SCA_SR_theta_ini,
#           SCA_SR_onoff_ini, P_Rmin, S_Rmin)

# =================================================================================================================== #
# -------------------------------------------------Scenario SE_RIS test---------------------------------------------- #
# =================================================================================================================== #

# SE_BB_RIS_power_ini = 20
# SE_BB_RIS_power_iniS = 20
# SE_BB_RIS_theta_ini = 0
# SE_BB_RIS_onoff_ini = 1

# # Time_RIS_BB = 0
# SE_BB_RIS_start = timeit.default_timer()

# SE_RIS_BB_status, SE_RIS_BB_objective, SE_RIS_power_BB, SE_RIS_theta_BB, SE_Time_RIS_BB = SE_RIS.SE_RIS_I(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, SE_BB_RIS_power_ini, SE_BB_RIS_power_iniS, SE_BB_RIS_theta_ini,
#           SE_BB_RIS_onoff_ini, P_Rmin_RIS, S_Rmin_RIS)

# SE_BB_RIS_end = timeit.default_timer()
# SE_Time_RIS_BB += (SE_BB_RIS_end - SE_BB_RIS_start)
# print("BB time use : ", SE_Time_RIS_BB)

# =================================================================================================================== #
# -------------------------------------------------Scenario SE_BD test---------------------------------------------- #
# =================================================================================================================== #

# BB_SEBD_power_ini = 40
# # BB_BD_power_iniS = 20
# BB_SEBD_theta_ini = 4
# BB_SEBD_onoff_ini = 1

# # Time_RIS_BB = 0
# BB_SEBD_start = timeit.default_timer()

# SEBD_BB_status, SEBD_BB_objective, SEBD_power_BB, SEBD_theta_BB, Time_SEBD_BB = SEBD.SE_SRBD(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           L_spreadingfactor, P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, BB_SEBD_power_ini,  BB_SEBD_theta_ini,
#           BB_SEBD_onoff_ini, P_Rmin, S_Rmin)

# BB_SEBD_end = timeit.default_timer()
# Time_SEBD_BB += (BB_SEBD_end - BB_SEBD_start)
# print("BB time use : ", Time_SEBD_BB)

# =================================================================================================================== #
# ------------------------------------------------Scenario SE_SR_BB test--------------------------------------------- #
# =================================================================================================================== #

# SE_BB_power_ini = 20
# SE_BB_theta_ini = 0
# SE_BB_onoff_ini = 1

# SE_BB_start = timeit.default_timer()

# SE_Symbiotic_BB_status, SE_Symbiotic_BB_objective, SE_power_BB, SE_theta_BB, SE_Time_BB = MultiU_BB_SE(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2, L_spreadingfactor,
#           P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, SE_BB_power_ini, SE_BB_theta_ini, SE_BB_onoff_ini, P_Rmin, S_Rmin)

# SE_BB_end = timeit.default_timer()
# SE_Time_BB += (SE_BB_end - SE_BB_start)
# print("BB time use : ", SE_Time_BB)

# =================================================================================================================== #
# -------------------------------------------------Scenario EE_RIS test---------------------------------------------- #
# =================================================================================================================== #

# BB_RIS_power_ini = 20
# BB_RIS_power_iniS = 20
# BB_RIS_theta_ini = 0
# BB_RIS_onoff_ini = 1

# # Time_RIS_BB = 0
# BB_RIS_start = timeit.default_timer()

# RIS_BB_status, RIS_BB_objective, RIS_power_BB, RIS_theta_BB,Time_RIS_BB = EE_RIS.EE_RIS_I(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, BB_RIS_power_ini, BB_RIS_power_iniS, BB_RIS_theta_ini,
#           BB_RIS_onoff_ini, P_Rmin_RIS, S_Rmin_RIS)

# BB_RIS_end = timeit.default_timer()
# Time_RIS_BB += (BB_RIS_end - BB_RIS_start)
# print("BB time use : ", Time_RIS_BB)


# =================================================================================================================== #
# ------------------------------------------------Scenario BD_SR test------------------------------------------------ #
# =================================================================================================================== #

# BB_BD_power_ini = 31
# # BB_BD_power_iniS = 20
# BB_BD_theta_ini = 0
# BB_BD_onoff_ini = 1

# # Time_RIS_BB = 0
# BB_BD_start = timeit.default_timer()

# BD_BB_status, BD_BB_objective, BD_power_BB, BD_theta_BB,Time_BD_BB = SRBD.EE_SRBD(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2,
#           L_spreadingfactor, P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, BB_BD_power_ini,  BB_BD_theta_ini,
#           BB_BD_onoff_ini, P_Rmin, S_Rmin)

# BB_BD_end = timeit.default_timer()
# Time_BD_BB += (BB_BD_end - BB_BD_start)
# print("BB time use : ", Time_BD_BB)

# =================================================================================================================== #
# --------------------------------------------------BB test--------------------------------------------------------- #
# =================================================================================================================== #

# BB_power_ini = 21
# BB_theta_ini = 0
# BB_onoff_ini = 1

# BB_start = timeit.default_timer()

# Symbiotic_BB_status, Symbiotic_BB_objective, power_BB, theta_BB, Time_BB = MultiU_BB(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, GroupRIS, Bandwidth, sigma2, L_spreadingfactor,
#           P_B2U_Ray, B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, BB_power_ini, BB_theta_ini, BB_onoff_ini, P_Rmin, S_Rmin)

# BB_end = timeit.default_timer()
# Time_BB += (BB_end - BB_start)
# print("BB time use : ", Time_BB)

#===================================================================================================================#

# phase_shift = (np.diag(np.exp(1j*(2*ma.pi*theta_BB/(8)))))

# Total_primary_channel = Funcom.Total_channel_conbime(phase_shift, B2R_Ray, P_R2U_Ray, P_B2U_Ray)
# Total_minus_channel = Funcom.Total_channel_minus(phase_shift, B2R_Ray, P_R2U_Ray, P_B2U_Ray)
# Total_secondary_channel = Funcom.Total_channel_conbime(phase_shift, B2R_Ray, S_R2U_Ray, S_B2U_Ray)

# Throughput_test_P = Funcom.primary_throughput(power_BB, Total_primary_channel, Total_minus_channel)
# Throughput_test_S = Funcom.secondary_throughput(power_BB, Total_secondary_channel, L_spreadingfactor)
# Power_consume = Funcom.power_consumption(power_BB, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static)

# Obj_func_test = Funcom.obj_func(Throughput_test_P, Throughput_test_S, Power_consume)

# print("Func throughput P : ", Throughput_test_P, "\nFunc throughput_ S : ", Throughput_test_S, "\nFunc power", Power_consume, "\nFunc obj : ", Obj_func_test)



# =================================================================================================================== #
# --------------------------------------------------Simple GA-------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 50
# mutate_rate = 0.2
# SGA_Max_itreation = 1000
# test_num = 30
# SGAObj_test_arr = np.zeros(test_num, dtype=np.float64)
# SGAtime_test_arr = np.zeros(test_num, dtype=np.float64)

# SGA = sga.Simple_Genetic(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                         P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                         B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray)

# for idx_sga in range(test_num):
    
#     SGAObj_test_arr[idx_sga], SGA_discrete_sol, SGA_best_phase, SGA_best_power, SGAtime_test_arr[idx_sga] = SGA.Simple_Gene_Algorithm(SGA_Max_itreation)
    
# print("SGA obj arr :\n", SGAObj_test_arr, "\nSGA Obj average : ", np.sum(SGAObj_test_arr)/test_num, "\nSGA time average : ", np.sum(SGAtime_test_arr)/test_num)



# SGA_Obj, SGA_discrete_sol, SGA_best_phase, SGA_best_power, SGA_time, SGA_eachiter_best = SGA.Simple_Gene_Algorithm(SGA_Max_itreation)
# print("SGA Obj : ", SGA_Obj, "\nSGA phase : ", SGA_best_phase, "\nSGA power : ", SGA_best_power, "\nSGA time : ", SGA_time)

# with open('ga_iter_output', 'w', newline='') as csvfile:

#     writer = csv.writer(csvfile)
#     for idx_wr in range(SGA_Max_itreation):
#         writer.writerow(str(SGA_eachiter_best[idx_wr]))
        
# =================================================================================================================== #
# --------------------------------------------------GA test--------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 50
# mutate_rate = 0.2
# GA_Max_itreation = 1000
# test_num = 15
# GAObj_test_arr = np.zeros(test_num, dtype=np.float64)
# GAtime_test_arr = np.zeros(test_num, dtype=np.float64)

# GAfunc = GA_init.Genetic(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                         P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                         B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray)

# for idx_t in range(test_num):
#     GAObj_test_arr[idx_t], GA_discrete_sol, GA_best_phase, GA_best_power, GAtime_test_arr[idx_t] = GAfunc.Gene_Algorithm(GA_Max_itreation)
    

# print("GA test :", test_num, "\n", GAObj_test_arr,"\nGA time :\n", GAtime_test_arr)
# print("GA average : ", np.sum(GAObj_test_arr)/test_num, "\n GA time Average :", np.sum(GAtime_test_arr)/test_num)


# GA_Obj, GA_discrete_sol, GA_best_phase, GA_best_power, GA_time, GA_iter_best = GAfunc.Gene_Algorithm(GA_Max_itreation)
# print("GA Obj : ", GA_Obj, "\nGA phase : ", GA_best_phase, "\nGA power : ", GA_best_power, "\nGA time : ", GA_time)

# population_test = GAfunc.Gen_Population()
# print("population :\n", population_test, "\nshape : ", population_test.shape)

# for i in population_test:
    
#     print("Gene : ", i, "\n")

# best_gene = population_test[3]

# phase_best = GAfunc.Phase_shift_decode(best_gene[0:Num_RISele])
# power_best = GAfunc.Power_decode(best_gene[Num_RISele:])
# # print("Phase : ", phase_best, "\nPhase gene : ", best_gene[0:Num_RISele], "\nPower : ", power_best, "\nPower gene : ", best_gene[Num_RISele:])

# fitness_test, temp_obj = GAfunc.fitness(best_gene)
# print("Fitness : ", fitness_test, "\nObj test : ", temp_obj)

# phase_best = np.ones(16)*(-1)
# power_best = np.array([0.01584893, 0.00251189, 0.63095734 , 0.25118864])









# Total_primary_channel = Funcom.Total_channel_conbime(np.diag(GA_best_phase), B2R_Ray, P_R2U_Ray, P_B2U_Ray)
# Total_minus_channel = Funcom.Total_channel_minus(np.diag(GA_best_phase), B2R_Ray, P_R2U_Ray, P_B2U_Ray)
# Total_secondary_channel = Funcom.Total_channel_conbime(np.diag(GA_best_phase), B2R_Ray, S_R2U_Ray, S_B2U_Ray)

# Throughput_test_P, Throughput_test_PUE = Funcom.primary_throughput(GA_best_power, Total_primary_channel, Total_minus_channel)
# Throughput_test_S, Throughput_test_SUE = Funcom.secondary_throughput(GA_best_power, Total_secondary_channel, L_spreadingfactor)
# Power_consume = Funcom.power_consumption(GA_best_power, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static)

# Obj_GA_veri = Funcom.obj_func(Throughput_test_P, Throughput_test_S, Power_consume)
# print("GA Obj V : ", Obj_GA_veri)
# for i in range(NumPUE):
#     print("Throughput UE P", Throughput_test_PUE[i])

# for i in range(NumSUE):
#     print("Throughput UE S", Throughput_test_SUE[i])

# =================================================================================================================== #
# --------------------------------------------------DWOA test--------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 50
# mutate_rate = 0.2
# Max_iteration = 300
# # whale_iteration = 5
# whale_a = 2
# whale_b = 1
# # whale_a_step = whale_a / whale_iteration
# # whale_a_step = whale_a / Max_iteration

# test_num = 30
# DWOAObj_test_arr = np.zeros(test_num, dtype=np.float64)
# DWOAtime_test_arr = np.zeros(test_num, dtype=np.float64)


# DWOA = dwoa.Discrete_whale_genetic(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                             P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                             B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b)


# DWOA_best_fitness, DWOA_best_sol, DWOA_time, DWOA_iter_best = DWOA.discrete_whale_genetic_algorithm(Max_iteration)
# # print("DWOA best fit : ", DWOA_best_fitness, "\nDWOA best solution : \n", DWOA_best_sol, "\n DWOA time :", DWOA_time)

# DWOA.discrete_whale_genetic_algorithm(Max_iteration)

# for idx_dwoa in range(test_num):
    
#     whale_a = 2
#     whale_b = 1
    
#     DWOA = dwoa.Discrete_whale_genetic(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                                 P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                                 B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b)
    
#     DWOAObj_test_arr[idx_dwoa], DWOA_best_sol, DWOAtime_test_arr[idx_dwoa] = DWOA.discrete_whale_genetic_algorithm(Max_iteration)

# print("DWOA obj arr :\n", DWOAObj_test_arr, "\nDWOA Obj average : ", np.sum(DWOAObj_test_arr)/test_num, "\nDWOA time average : ", np.sum(DWOAtime_test_arr)/test_num)

# =================================================================================================================== #
# --------------------------------------------------GWOA test--------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 30
# mutate_rate = 0.25
# Max_iteration = 800
# whale_iteration = 5
# whale_a = 0.7
# whale_b = 0.5
# whale_a_step = whale_a / whale_iteration
# # whale_a_step = whale_a / Max_iteration


# GWOA = gwoa.Genetic_whale(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                         P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                         B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b, whale_a_step)

# GWOA_obj, GWOA_best_sol, GWOA_best_phase, GWOA_best_power,GWOA_time = GWOA.Gene_whale_Algorithm(Max_iteration, whale_iteration)
# print("GWOA Obj : ", GWOA_obj, "\nGWOA sol : ", GWOA_best_sol, "\nGA time : ", GWOA_time)

# =================================================================================================================== #
# --------------------------------------------------WOGA test--------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 50
# mutate_rate = 0.2
# Max_iteration = 6
# whale_iteration = 20
# whale_step_num = 100
# ga_iteration = 70
# whale_a = 1.2
# whale_b = 0.5
# # whale_a_step = whale_a / whale_iteration
# whale_a_step = whale_a / whale_step_num

# test_num = 15
# WOGAObj_test_arr = np.zeros(test_num, dtype=np.float64)
# WOGAtime_test_arr = np.zeros(test_num, dtype=np.float64)


# for idx_t in range(test_num):
    
#     WOGA = woga.Whale_genetic(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                             P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                             B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b, whale_a_step)

    
#     # print("!!! New problem !!! : ", idx_t)
#     WOGAObj_test_arr[idx_t], WOGA_best_sol, WOGAtime_test_arr[idx_t] = WOGA.whale_genetic_algorithm(Max_iteration, whale_iteration, ga_iteration)
# print("WOGA test :", test_num, "\n", WOGAObj_test_arr,"\nWOGA time :\n", WOGAtime_test_arr)
# print("WOGA Average : ", np.sum(WOGAObj_test_arr)/test_num, "\nWOGA time Average : ", np.sum(WOGAtime_test_arr)/test_num)


# WOGA_best_fitness, WOGA_best_sol, WOGA_time = WOGA.whale_genetic_algorithm(Max_iteration, whale_iteration, ga_iteration)
# print("WOGA best fit : ", WOGA_best_fitness, "\nWOGA best solution : \n", WOGA_best_sol, "\n WOGA time :", WOGA_time)

# =================================================================================================================== #
# --------------------------------------------------WOA test--------------------------------------------------------- #
# =================================================================================================================== #

# population_size = 250
# mutate_rate = 0.2
# WOA_Max_itreation = 300
# whale_a = 2
# whale_b = 2.5
# whale_a_step = whale_a / WOA_Max_itreation

# test_num = 30
# WOAObj_test_arr = np.zeros(test_num, dtype=np.float64)
# WOAtime_test_arr = np.zeros(test_num, dtype=np.float64)

# for idx_woa in range(test_num):
    
#     whale_a = 2
#     whale_b = 2.5
#     whale_a_step = whale_a / WOA_Max_itreation
#     WOA = woa.Whale(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                             P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                             B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b, whale_a_step)
    
    
#     WOAObj_test_arr[idx_woa], whale_best_sol, WOAtime_test_arr[idx_woa] = WOA.whale_algorithm(WOA_Max_itreation)

# print("WOA test :", test_num, "\n", WOAObj_test_arr, "\nWOA time :\n", WOAtime_test_arr)
# print("WOA Average : ", np.sum(WOAObj_test_arr)/test_num, "\nWOA time Average : ", np.sum(WOAtime_test_arr)/test_num)



# WOA = woa.Whale(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, 
#                             P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                             B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray, whale_a, whale_b, whale_a_step)


# whale_best_fitness, whale_best_sol, WOA_time, WOA_iter_best = WOA.whale_algorithm(WOA_Max_itreation)
# print("WOA best fit : ", whale_best_fitness, "\nWOA best solution : \n", whale_best_sol, "\nWOA best solution : ", WOA_time)

# =================================================================================================================== #
# --------------------------------------------------EXH test--------------------------------------------------------- #
# =================================================================================================================== #

# Exhsearch = Exh.exhaustive(NumBSAnt, Num_RISele, NumUE, NumPUE, NumSUE, sigma2, RIS_bits, L_spreadingfactor, 
#                         P_Rmin, S_Rmin, P_max_o, PT_static, mu, SIC_dissapation, RIS_static, PUE_static, SUE_static,
#                         B2R_Ray, P_B2U_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray)

# Num_Variables = 20
# vector_varibles = np.zeros(Num_Variables, dtype=np.float128)
# Temp_best = -100
# power_gap = 4

# Exh_Obj, Exh_vector = Exhsearch.find_all_passwords(Num_Variables, Temp_best, vector_varibles, power_gap)

# print("Obj :", Exh_Obj,"\nVector", Exh_vector)




# =================================================================================================================== #
# --------------------------------------------------- plot ---------------------------------------------------------- #
# =================================================================================================================== #

# ============================================RIS element plot=============================================== #

# test_num_x2 = np.array([4, 8, 16, 24, 32])


# BB_avg_result = np.array([2.50028973922881, 2.59494269096299, 2.59730859965684, 2.61020995905615, 2.61385514437899])
# GA_avg_result = np.array([2.32200609355346, 2.46975208348651, 2.45493177602294,2.45609358009845, 2.39903625375072])
# WOA_avg_result = np.array([2.38387389162009, 2.5494146472795, 2.54516256199476, 2.58569091509221, 2.58031948415573])
# WOGA_avg_result = np.array([2.49760434391737, 2.59365178485412, 2.59373709084527, 2.60690665195655, 2.61091622234771])



# plt.plot(test_num_x2, BB_avg_result, 'rx--')
# plt.plot(test_num_x2, WOGA_avg_result, 'b.--')
# plt.plot(test_num_x2,  WOA_avg_result, 'g*--')
# plt.plot(test_num_x2,  GA_avg_result, 'yo--')

# plt.xlabel('RIS element number N')
# plt.ylabel('EE bit/J')
# plt.legend(('BB', 'WGA', 'WOA', 'GA'), loc='lower left')
# # plt.legend(('BB', 'WOA-based GA', 'WOA'), loc='lower left')


# plt.grid(True)
# plt.show()

# ============================================Excecution time plot=============================================== #

# time_label = np.array(['GA', 'WGA', 'WOA', 'BB'])

# Exe_time_result = np.array([58.3410826910908, 7.58379614185542, 9.38142889173081, 100.486472881399])

# plt.bar(time_label[0], Exe_time_result[0], color = 'k', width = 0.4, bottom = 2.3)
# plt.bar(time_label[1], Exe_time_result[1], color = 'c', width = 0.4, bottom = 2.3)
# plt.bar(time_label[2], Exe_time_result[2], color = 'b', width = 0.4, bottom = 2.3)
# plt.bar(time_label[3], Exe_time_result[3], color = 'g', width = 0.4, bottom = 2.3)

# plt.xlabel('Algorithm',)
# plt.ylabel('CPU execution time s')
# plt.legend(('GA', 'WGA', 'WOA', 'BB'), loc='upper left')

# plt.show()
# ============================================L plot=============================================== #

# ---Adjust secondary throyghput--- #

# L_num = np.array([10, 15, 20, 25, 30])

# S_Rmin2_result = np.array([2.599442572334, 2.5045396590645, 2.4525468588, 2.420611639279, 2.398915296757])
# S_Rmin4_result = np.array([2.59964389042916, 2.50430972831592, 2.45297096460853, 2.42208219348702, 2.37660485710914])
# S_Rmin5_result = np.array([2.59987760412891, 2.50457445395835, 2.4524386676857, 2.27075047852148, 1.25247511366724])
# S_Rmin45_result = np.array([2.59902911365573, 2.50440635505858, 2.45298916503565, 2.41910573159953, 1.98203973245427])

# plt.plot(L_num, S_Rmin2_result, 'rx--')
# plt.plot(L_num, S_Rmin4_result, 'b.--')
# plt.plot(L_num, S_Rmin45_result, 'yo--')
# plt.plot(L_num, S_Rmin5_result, 'g*--')

# plt.xlabel('Spreading factor L')
# plt.ylabel('EE bit/J')
# plt.legend(('Rs = 0.2 bps/Hz', 'Rs = 0.4 bps/Hz', 'Rs = 0.45 bps/Hz', 'Rs = 0.5 bps/Hz'), loc='lower left')

# x_locator = MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()

# ---secondary power--- #

# L_num = np.array([10, 15, 20, 25, 30])

# S_power2_result = np.array([2.00474893450909, 2.00474893450909, 2.00474893450909, 2.00474893450909, 2.00474893450909])
# S_power4_result = np.array([2.00474893450909, 2.00474893450909, 2.00474893450909, 2.00474893450909, 2.68720026816486])
# S_power45_result = np.array([2.00474893450909, 2.00474893450909, 2.00474893450909, 2.13451904536201, 7.26656600889292])
# S_power5_result = np.array([2.00474893450909, 2.00474893450909, 2.00474893450909, 4.25892541179417, 20.3143898228824])

# plt.plot(L_num, S_power2_result, 'rx--')
# plt.plot(L_num, S_power4_result, 'b.--')
# plt.plot(L_num, S_power45_result, 'yo--')
# plt.plot(L_num, S_power5_result, 'g*--')

# plt.xlabel('Spreading factor L',)
# plt.ylabel('Power Watt')
# plt.legend(('Rs = 0.2 bps/Hz', 'Rs = 0.4 bps/Hz', 'Rs = 0.45 bps/Hz', 'Rs = 0.5 bps/Hz'), loc='upper left')

# x_locator = MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()





# ---Adjust primary throyghput--- #

# L_num = np.array([10, 15, 20, 25, 30])

# P_Rmin7_result = np.array([2.53082222729663, 2.43780876994441, 2.38699293351166, 2.25102818878514, 1.25247511366724])
# P_Rmin8_result = np.array([2.30503887598359, 2.22234576365593, 2.17813420619578, 2.13670006085713, 1.25247511366724])
# P_Rmin9_result = np.array([1.86177710923143, 1.7961741534792, 1.76138673501513, 1.73999904705062, 1.25247511366724])
# P_Rmin10_result = np.array([1.3250328545846, 1.27892207778113, 1.25444158854135, 1.23923796715928, 1.12871125690705])

# plt.plot(L_num, P_Rmin7_result, 'rx--')
# plt.plot(L_num, P_Rmin8_result, 'b.--')
# plt.plot(L_num, P_Rmin9_result, 'yo--')
# plt.plot(L_num, P_Rmin10_result, 'g*--')

# plt.xlabel('Spreading factor L',)
# plt.ylabel('EE bit/J')
# plt.legend(('Rp = 7 bps/Hz', 'Rp = 8 bps/Hz', 'Rp = 9 bps/Hz', 'Rp = 10 bps/Hz'), loc='upper right')

# x_locator = MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()


# ---primary power--- #

# L_num = np.array([10, 15, 20, 25, 30])

# P_power7_result = np.array([2.76248711267598, 2.76248711267598, 2.76248711267598, 4.3792214271854, 20.3143898228824])
# P_power8_result = np.array([5.14284377598977, 5.14284377598977, 5.14284377598977, 5.30621466623386, 20.3143898228824])
# P_power9_result = np.array([10.2613223780047, 10.2613223780047, 10.2613223780047, 10.2613223780047, 20.3143898228824])
# P_power10_result = np.array([20.4740298425795, 20.4740298425795, 20.4740298425795, 20.4740298425795, 24.0048163780804])

# plt.plot(L_num, P_power7_result, 'rx--')
# plt.plot(L_num, P_power8_result, 'b.--')
# plt.plot(L_num, P_power9_result, 'yo--')
# plt.plot(L_num, P_power10_result, 'g*--')

# plt.xlabel('Spreading factor L',)
# plt.ylabel('Power Watt')
# plt.legend(('Rp = 7 bps/Hz', 'Rp = 8 bps/Hz', 'Rp = 9 bps/Hz', 'Rp = 10 bps/Hz'), loc='upper left')

# x_locator = MultipleLocator(5)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()


# ============================================Number of user=============================================== #


# User_num = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# User_BB_result = np.array([0.853883102952923, 1.60082099300254, 2.15587528496648, 2.59730859965684, 2.96011232140721, 3.34905852625529, 3.63110256991411, 3.73389738182651])
# User_GA_result = np.array([0.845708880297368, 1.58917903849284, 2.11616079631586, 2.37110920971029, 2.51666537536627, 2.7377260341558, 2.82209664094632, 2.57161870852522])
# User_WOA_result = np.array([0.851376493496351, 1.59752227232441, 2.14897924602529, 2.53865591890766, 2.91680128326362, 3.29820644568059, 3.5733148481908, 3.63198565450316])
# User_DWOA_result = np.array([0.853334269977819, 1.60041118304999, 2.15522070751845, 2.59373709084527, 2.95258935700283, 3.34481517571145, 3.61620678306028, 3.72657489443402])

# User_num = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# User_BB_result = np.array([0.853883102952923, 1.60082099300254, 2.15587528496648, 2.59730859965684, 2.96011232140721, 3.34905852625529, 3.63110256991411, 3.73389738182651])
# User_GA_result = np.array([0.845708880297368, 1.58917903849284, 2.11616079631586, 2.37110920971029, 2.51666537536627, 2.7377260341558, 2.82209664094632, 2.57161870852522])
# User_WOA_result = np.array([0.850222090611938, 1.59673725351705, 2.146220348080766, 2.57028906266239, 2.90310120935861, 3.3014721791283, 3.53645689972099, 3.52413233367319])
# User_DWOA_result = np.array([0.853334269977819, 1.60041118304999, 2.15522070751845, 2.59373709084527, 2.95258935700283, 3.34481517571145, 3.61620678306028, 3.72657489443402])

# ---Different user--- #

# User_num = np.array([4, 5, 6, 7, 8])

# User_BB_result = np.array([ 2.59730859965684, 2.96011232140721, 3.34905852625529, 3.63110256991411, 3.73389738182651])
# User_GA_result = np.array([2.37110920971029, 2.51666537536627, 2.7377260341558, 2.82209664094632, 2.57161870852522])
# User_WOA_result = np.array([ 2.53865591890766, 2.91680128326362, 3.29820644568059, 3.5733148481908, 3.63198565450316])
# User_DWOA_result = np.array([2.59373709084527, 2.95258935700283, 3.34481517571145, 3.61620678306028, 3.72657489443402])

# plt.plot(User_num, User_BB_result, 'rx--')
# plt.plot(User_num, User_GA_result, 'b.--')
# plt.plot(User_num, User_WOA_result, 'yo--')
# plt.plot(User_num, User_DWOA_result, 'g*--')



# plt.xlabel('Number of user',)
# plt.ylabel('EE bit/J')
# plt.legend(('BB', 'GA', 'WOA', 'WGA'), loc='upper left')
# # plt.legend(('BB', 'WOA', 'WOA-based GA'), loc='upper left')

# x_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()

# ---Different seondary user--- #

# User_num = np.array([4, 5, 6, 7, 8])

# User_P2_result = np.array([1.97250348149569, 2.0805766775076, 2.16827385481068, 2.26420341155526, 2.3500901594801])
# User_P3_result = np.array([2.53100738146857, 2.65547589386334, 2.73912361891412, 2.82855269610482, 2.90243527900944])
# User_P4_result = np.array([2.96967310038186, 3.04497703752915, 3.12916593403314, 3.23203348198792, 3.23400039529082])
# # User_DWOA_result = np.array([2.59373709084527, 2.95258935700283, 3.34481517571145, 3.61620678306028, 3.72657489443402])

# plt.plot(User_num, User_P2_result, 'rx--')
# plt.plot(User_num, User_P3_result, 'b.--')
# plt.plot(User_num, User_P4_result, 'yo--')
# # plt.plot(User_num, User_DWOA_result, 'g*--')



# plt.xlabel('Number of secondary user',)
# plt.ylabel('EE bit/J')
# plt.legend(('2-PRx', '3-PRx', '4-PRx'), loc='upper left')
# # plt.legend(('BB', 'WOA', 'WOA-based GA'), loc='upper left')

# x_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()

# ============================================Tteration Convergence=============================================== #

# Iter_num_sim = 800
# Iteration = np.array([idx_itr for idx_itr in range(Iter_num_sim )])

# GA_iteration_result = np.zeros(Iter_num_sim)
# WOA_iteration_result = np.zeros(Iter_num_sim)
# DWOA_iteration_result = np.zeros(Iter_num_sim)

# temp = ' '
# idx_count = 0

# with open('ga_iter_output3', newline='') as csvfile:
#     rows = csv.reader(csvfile)
    
#     for row in rows:
#         temp = ' '
#         for idx_c in range(len(row)):
#             temp += row[idx_c] 
        
#         # print(temp)
#         GA_iteration_result[idx_count] = float(temp)
#         idx_count += 1

# idx_count = 0
# with open('woa_iter_output3', newline='') as csvfile:
#     rows = csv.reader(csvfile)
    
#     for row in rows:
#         temp = ' '
#         for idx_c in range(len(row)):
#             temp += row[idx_c] 
        
#         # print(temp)
#         WOA_iteration_result[idx_count] = float(temp)
#         idx_count += 1

# idx_count = 0
# with open('dwoa_iter_output', newline='') as csvfile:
#     rows = csv.reader(csvfile)
    
#     for row in rows:
#         temp = ' '
#         for idx_c in range(len(row)):
#             temp += row[idx_c] 
        
#         # print(temp)
#         DWOA_iteration_result[idx_count] = float(temp)
#         idx_count += 1

# plt.plot(Iteration, GA_iteration_result, 'rx--')
# plt.plot(Iteration, WOA_iteration_result, 'yo--')
# plt.plot(Iteration, DWOA_iteration_result, 'g*--')

# plt.xlabel('Iteration',)
# plt.ylabel('EE bit/J')
# plt.legend('GA', loc='lower right')
# plt.legend(('GA', 'WOA', 'WGA'), loc='lower right')

# plt.grid(True)
# plt.show()

# ============================================Scenario Compare=============================================== #

# ------EE------ #

# User_num_scenario = np.array([1, 2, 3, 4, 5])

# EE_RISSR_result = np.array([ 0.853883102952923, 1.60082099300254, 2.15587528496648, 2.59730859965684, 2.96011232140721])
# EE_BDSR_result = np.array([ 0.837983076994936, 1.57794951613686, 2.10283635676877, 2.46858990266661, 2.74437665628038])
# EE_RIS_result = np.array([ 0.425673605079747, 0.286966637181786, 0.2282329493116, 0.1872329493116, 0.1562329493116])

# plt.plot(User_num_scenario , EE_RISSR_result, 'rx--')
# plt.plot(User_num_scenario , EE_BDSR_result, 'b.--')
# plt.plot(User_num_scenario, EE_RIS_result, 'yo--')
# # plt.plot(User_num, User_DWOA_result, 'g*--')

# plt.xlabel('Number of user',)
# plt.ylabel('EE bit/J')
# plt.legend(('RIS-SR', 'BD-SR', 'RIS-assisted'), loc='upper left')

# x_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_locator)

# plt.grid(True)
# plt.show()

# ------SE------ #

User_num_scenario = np.array([1, 2, 3, 4, 5])

SE_RISSR_result = np.array([ 13.6472601312416, 26.4887086815345, 38.2157484809043, 48.8013748540434, 59.2718706901948])
SE_BDSR_result = np.array([ 9.86101142530721, 21.5818818962006, 32.4409973004514, 42.956897897128, 53.2275765208822])
SE_RIS_result = np.array([ 3.87198132392908, 2.46162623349715, 1.95208026557746, 1.71566639892354, 1.59988769393111])

plt.plot(User_num_scenario , SE_RISSR_result, 'rx--')
plt.plot(User_num_scenario , SE_BDSR_result, 'b.--')
plt.plot(User_num_scenario, SE_RIS_result, 'yo--')
# plt.plot(User_num, User_DWOA_result, 'g*--')

plt.xlabel('Number of user',)
plt.ylabel('SE bps/Hz')
plt.legend(('RIS-SR', 'BD-SR', 'RIS-assisted'), loc='upper left')

x_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_locator)

plt.grid(True)
plt.show()
