#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:31:26 2023

@author: dddd
"""

import numpy as np
import pandas as pd
import math as ma
import time
from pyomo.environ import *

import cvxpy as cvx

def SR_SU(NumBSAnt, NumRISEle, NumUE, NumPUE, NumSUE, RISgroup, Bandwidth, sigma2, L_spreadingfactor,
              H_U2B_Ric, H_R2B_Ric, H_U2R_Ric, S_U2B_Ric, S_U2R_Ric, BB_power_ini, BB_theta_ini, BB_onoff_ini, P_Rmin, S_Rmin, SUE_times):
    
    # np.random.seed()
    # 0 for Primary
    # 1 for Secondary
    # NumSUE = SUE_times*NumSUE
    mu=1.2
    PUE_static = 10  #dBm
    PUE_static = pow(10, (PUE_static/10))/pow(10,3)
    
    SUE_static = 10  #dBm
    SUE_static = pow(10, (SUE_static/10))/pow(10,3)
    
    PT_static = 39  #dBm
    PT_static = pow(10, (PT_static/10))/pow(10,3)
    
    RIS_static = 10  #dBm
    RIS_static = pow(10, (RIS_static/10))/pow(10,3)
    
    SIC_dissapation = 0.2
    
    # PUE_P = [35, 35, 35, 35]
    # # PUE_power = np.zeros([NumPUE, 1], dtype = float)
    # PUE_power = []
    # for idx_PUE in range(NumPUE):
    #     PUE_power.append(pow(10, (PUE_P[idx_PUE]/10))/pow(10,3))
    
    P_max = 45 #dbm 

    Group_UE = int(NumUE/RISgroup)
    
    T_Conbime = np.zeros([RISgroup, NumBSAnt*NumSUE, NumRISEle], dtype = 'complex_')
    RIS_link_real = np.zeros([RISgroup, Group_UE, NumBSAnt], dtype = 'complex_')
    RIS_link_imag = np.zeros([RISgroup, Group_UE, NumBSAnt], dtype = 'complex_')
    
    Received_channel_real = np.zeros([RISgroup, Group_UE, NumBSAnt], dtype = 'complex_')
    Received_channel_imag = np.zeros([RISgroup, Group_UE, NumBSAnt], dtype = 'complex_')
    
    H_U2B_RicRe = np.zeros([RISgroup, NumSUE, NumBSAnt], dtype = 'complex_')
    
    # flag_B2U = 0
    # for idx_totalUE in range(NumUE):
    #     if (int(idx_totalUE % Group_UE) == 0 and idx_totalUE != 0):
    #         flag_B2U += 1
    #     H_U2B_RicRe[flag_B2U][int(idx_totalUE % Group_UE), :] = H_U2B_Ric[idx_totalUE,:]
    
    H_U2B_RicRe[0][0:NumPUE] = H_U2B_Ric
    H_U2B_RicRe[1] = S_U2B_Ric   
    
    # print("H_B2U Orinal : \n", H_U2B_Ric)
    # print("H_B2U 0 : \n", H_U2B_RicRe[0], "\nH_B2U 1 : \n", H_U2B_RicRe[1])
    
    Phase_shift_test = np.ones([RISgroup * NumRISEle, 1], dtype = 'float_') * (-ma.pi/4)
    Phase_shift_test_0 = np.ones([NumRISEle, 1], dtype = 'float_') * (-ma.pi/4)
    Phase_shift_test_1 = np.ones([NumRISEle, 1], dtype = 'float_') * (-ma.pi/5)
    # for idx_pha in range(NumRISEle):
    #     Phase_shift_test[idx_pha + (NumSUE-1)*NumRISEle] = Phase_shift_test_1[idx_pha]
    
    # xonoff = np.array([0,1])
    
    PUE_S = 30
    SUE_power = []
    for idx_PUE in range(NumSUE):
        SUE_power.append(pow(10, (PUE_S/10))/pow(10,3))
        
        
    # channel_square = np.ones([NumUE, 1], dtype = 'complex_')
    
    flag_row_1 = 0
    flag_row_2 = 0
    for idx_B2R_i in range(0, NumBSAnt*int(NumPUE)):
        # idx_R2B_j = 0
        if(flag_row_1 == NumBSAnt):
            flag_row_1 = 0
            flag_row_2 += 1 #+ (idx_group)
        
        for idx_RISEle in range(0, NumRISEle):
            
            T_Conbime[0][idx_B2R_i][idx_RISEle] = H_U2R_Ric[flag_row_2][idx_RISEle]*H_R2B_Ric[idx_RISEle][(flag_row_1)]
            
            # print("Row flag : ", int(idx_R2B_i/4))
        
        flag_row_1 += 1
        
    flag_row_1 = 0
    flag_row_2 = 0
    for idx_B2R_i in range(0, NumBSAnt*int(NumSUE)):
        # idx_R2B_j = 0
        if(flag_row_1 == NumBSAnt):
            flag_row_1 = 0
            flag_row_2 += 1 #+ (idx_group)
        
        for idx_RISEle in range(0, NumRISEle):

            T_Conbime[1][idx_B2R_i][idx_RISEle] = S_U2R_Ric[flag_row_2][idx_RISEle]*H_R2B_Ric[idx_RISEle][(flag_row_1)]
            
            # print("Row flag : ", int(idx_R2B_i/4))
        
        flag_row_1 += 1
    
    # print("T_Conbime_P : \n", T_Conbime[0], "\n", "T_Conbime_S : \n", T_Conbime[1])
    T_0 = T_Conbime[0] @ Phase_shift_test_0
    T_1 = T_Conbime[1] @ Phase_shift_test_1
    
    
    T_temp_1 = np.zeros([Group_UE, NumRISEle], dtype = 'complex_')
    T_temp_2 = np.zeros([Group_UE, NumRISEle], dtype = 'complex_')
    
    # T_temp_1[0,:] = H_U2R_Ric[0,:]
    # T_temp_1[1,:] = H_U2R_Ric[1,:]
    # T_temp_2[0,:] = H_U2R_Ric[2,:]
    # T_temp_2[1,:] = H_U2R_Ric[3,:]
    
    # print(T_temp_1.shape)
    # print("B2R : \n", H_R2B_Ric.shape)
    # print("Multiply : \n", (T_temp_1 @ np.eye(NumRISEle) * Phase_shift_test_1).shape)
    T_veri_0 = H_U2R_Ric @ (np.eye(NumRISEle) * Phase_shift_test_0) @ H_R2B_Ric
    T_veri_1 = S_U2R_Ric @ (np.eye(NumRISEle) * Phase_shift_test_1) @ H_R2B_Ric
    
    # print("T_veri_0 \n: ", T_veri_0)
    # print("T_0 \n: ", T_0.shape, "T_0 \n: ", T_0)
    
    # print("T_veri_1 \n: ", T_veri_1)
    # print("T_1 \n: ", T_1.shape,"T_1 \n: ", T_1)
    
    
    
    
    # flag_real = 0
    # for idx_rflag in range(0, NumBSAnt):
    #     for idx_real in range(0, NumUE):
    #         RIS_link_real[idx_rflag][idx_real] = sum((T_Conbime[flag_real][idx_ele].real*ma.cos(Phase_shift_test[idx_ele]) - T_Conbime[flag_real][idx_ele].imag*ma.sin(Phase_shift_test[idx_ele])) for idx_ele in range(0, NumRISEle))
    #         flag_real += 1
    
    # # print("Real flag : ", flag_real)
            
    # flag_imag = 0
    # for idx_iflag in range(0, NumBSAnt):
    #     for idx_imag in range(0, NumUE):
    #         RIS_link_imag[idx_iflag][idx_imag] = sum((T_Conbime[flag_imag][idx_ele].real*ma.sin(Phase_shift_test[idx_ele]) + T_Conbime[flag_imag][idx_ele].imag*ma.cos(Phase_shift_test[idx_ele]))  for idx_ele in range(0, NumRISEle))
    #         flag_imag += 1
            
    # for idx_re_BS in range(0, NumBSAnt):
    #     for idx_re_UE in range(0, NumUE):
    #         Received_channel_real[idx_re_BS][idx_re_UE] = H_U2B_Ric[idx_re_BS][idx_re_UE].real + RIS_link_real[idx_re_BS][idx_re_UE]
    #         Received_channel_imag[idx_re_BS][idx_re_UE] = H_U2B_Ric[idx_re_BS][idx_re_UE].imag + RIS_link_imag[idx_re_BS][idx_re_UE]
            
    # for idx_UE in range(0, NumUE):
    #     channel_square[idx_UE] = sum(Received_channel_real[idx_BS][idx_UE]**2 +Received_channel_imag[idx_BS][idx_UE]**2 for idx_BS in range(0, NumBSAnt))
    
    
    
    
    
    flag_channel_finish = "True"
    
    #=====================================================Channel Test============================================================#
    
    # def RIS_received_real_channel(idx_realg, idx_realcha):
        
    #     # return sum((T_Conbime[idx_risreal][idx_ele].real*cos(model.theta[idx_ele]) - T_Conbime[idx_risreal][idx_ele].imag*sin(model.theta[idx_ele])) for idx_ele in range(0, NumRISEle))
    #     return sum((T_Conbime[idx_realg][idx_realcha][idx_ele].real*cos(Phase_shift_test_1[idx_ele]) - T_Conbime[idx_realg][idx_realcha][idx_ele].imag*sin(Phase_shift_test_1[idx_ele])) for idx_ele in range(0, NumRISEle))
        
    # def RIS_received_imag_channel(idx_imagg, idx_imagcha):
        
    #     # return sum((T_Conbime[idx_risimag][idx_ele].real*sin(model.theta[idx_ele]) + T_Conbime[idx_risimag][idx_ele].imag*cos(model.theta[idx_ele]))  for idx_ele in range(0, NumRISEle))
    #     return sum((T_Conbime[idx_imagg][idx_imagcha][idx_ele].real*sin(Phase_shift_test_1[idx_ele]) + T_Conbime[idx_imagg][idx_imagcha][idx_ele].imag*cos(Phase_shift_test_1[idx_ele]))  for idx_ele in range(0, NumRISEle))
    
    # def RIS_square_channel(idx_group, idx_square):
        
    #     # return sum(Received_channel_real[idx_BS][idx_UE]**2 +Received_channel_imag[idx_BS][idx_UE]**2 for idx_BS in range(0, NumBSAnt))
        
    #     return sum((H_U2B_RicRe[idx_group][idx_square][idx_BS].real +RIS_received_real_channel(idx_group, idx_square*NumBSAnt + idx_BS))**2 +\
    #                 (H_U2B_RicRe[idx_group][idx_square][idx_BS].imag +RIS_received_imag_channel(idx_group, idx_square*NumBSAnt + idx_BS))**2 for idx_BS in range(0, NumBSAnt))
    
    # Channel_total_veri_0 = H_U2B_RicRe[0] + H_U2R_Ric @ (np.eye(NumRISEle) *  np.exp(1j*Phase_shift_test_1)) @ H_R2B_Ric
    # Channel_total_veri_1 = H_U2B_RicRe[1] + S_U2R_Ric @ (np.eye(NumRISEle) *  np.exp(1j*Phase_shift_test_1)) @ H_R2B_Ric
    
    # Channel_square_veri_0 = np.diag(Channel_total_veri_0 @ Channel_total_veri_0.conj().T)
    # Channel_square_veri_1 = np.diag(Channel_total_veri_1 @ Channel_total_veri_1.conj().T)
    
    # Combine_channel_square = np.zeros(NumUE)
    # flag_square = 0
    # for idx_g in range(RISgroup):
    #     for idx_v in range(Group_UE):
    #         Combine_channel_square[flag_square] = RIS_square_channel(idx_g, idx_v)
    #         flag_square += 1
    
    # print("Channel total 0: \n", Channel_total_veri_0)
    # print("Channel total 1: \n", Channel_total_veri_1)
    # print("Channel square veri 0 :\n", Channel_square_veri_0, "\nChannel square veri 1 :\n", Channel_square_veri_1)
    # print("Combine channel square \n", Combine_channel_square)
    
    # def phase_shift_discrete_channel(idx_phase_cha):
        
    #     return (2*ma.pi*theta_integers[idx_phase_cha])/(2**NumUE)
    poewr_times = 1
    
    #----------------------------------------------------Solver model-------------------------------------------------------------#
    
    def obj_function(model, L_spreadingfactor, PT_static, RIS_static, SIC_dissapation, PUE_static, SUE_static):
        
        # return sum((Bandwidth*(log(1+((SUE_power[idx_UE_obj] * RIS_square(model, 1, idx_UE_obj)) / (interference(model, idx_UE_obj) + sigma2))) / log(2) ))/(SUE_power[idx_UE_obj]*mu +P_k) for idx_UE_obj in range(NumSUE))
        # return sum((Bandwidth*(log(1+((model.power[idx_UE_obj] * RIS_square(model, 1, idx_UE_obj)) / (interference(model, idx_UE_obj) + sigma2))) / log(2) ))/(model.power[idx_UE_obj]*mu +P_k) for idx_UE_obj in range(NumSUE))
        # return sum((0.5*((log(1 + ((model.power[idx_UE_obj] * Channel_square(model, 1, idx_UE_obj)) / sigma2)) / log(2) ) + \
        #             (log(1 + ((model.power[idx_UE_obj] * Channel_square_PUE(model, 1, idx_UE_obj)) / sigma2)) / log(2) )) )/(model.power[idx_UE_obj]*mu + P_k) for idx_UE_obj in range(NumSUE))
        return ( throughput_PUE(model) + throughput_SUE(model, L_spreadingfactor) ) / power_consumption(model, PT_static, RIS_static, SIC_dissapation, PUE_static, SUE_static)
        
        
    def RIS_received_real(model, idx_realg, idx_risreal):
        
        # return sum((T_Conbime[idx_realg][idx_risreal][idx_ele].real*cos(Phase_shift_test[idx_ele + idx_sur*NumRISEle]) - \
        #             T_Conbime[idx_realg][idx_risreal][idx_ele].imag*sin(Phase_shift_test[idx_ele + idx_sur*NumRISEle])) for idx_ele in range(0, NumRISEle))
        # return sum((T_Conbime[idx_realg][idx_risreal][idx_ele].real*cos(model.theta[idx_ele]) - \
        #             T_Conbime[idx_realg][idx_risreal][idx_ele].imag*sin(model.theta[idx_ele])) for idx_ele in range(0, NumRISEle))
        return sum((T_Conbime[idx_realg][idx_risreal][idx_ele].real*cos(phase_shift_discrete(model, idx_ele)) - \
                    T_Conbime[idx_realg][idx_risreal][idx_ele].imag*sin(phase_shift_discrete(model, idx_ele))) for idx_ele in range(0, NumRISEle))
        
    def RIS_received_imag(model, idx_imagg, idx_risimag):
        
        # return sum((T_Conbime[idx_imagg][idx_risimag][idx_ele].real*sin(Phase_shift_test[idx_ele + idx_sui*NumRISEle]) + \
        #             T_Conbime[idx_imagg][idx_risimag][idx_ele].imag*cos(Phase_shift_test[idx_ele + idx_sui*NumRISEle] ))  for idx_ele in range(0, NumRISEle))
        # return sum((T_Conbime[idx_imagg][idx_risimag][idx_ele].real*sin(model.theta[idx_ele]) + \
        #             T_Conbime[idx_imagg][idx_risimag][idx_ele].imag*cos(model.theta[idx_ele] ))  for idx_ele in range(0, NumRISEle))
        return sum((T_Conbime[idx_imagg][idx_risimag][idx_ele].real*sin(phase_shift_discrete(model, idx_ele)) + \
                    T_Conbime[idx_imagg][idx_risimag][idx_ele].imag*cos(phase_shift_discrete(model, idx_ele) ))  for idx_ele in range(0, NumRISEle))
    
    
    def Channel_square(model, idx_group, idx_square_UE): # wiith onoff
        
        # return sum((H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].real + xonoff[idx_square_UE] * RIS_received_real(model, idx_group, idx_square_UE, (idx_square_UE*NumBSAnt + idx_BS)))**2 +\
        #            (H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].imag + xonoff[idx_square_UE] * RIS_received_imag(model, idx_group, idx_square_UE, (idx_square_UE*NumBSAnt + idx_BS)))**2 for idx_BS in range(0, NumBSAnt))
        return sum((H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].real +  RIS_received_real(model, idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 +\
                    (H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].imag + RIS_received_imag(model, idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 for idx_BS in range(0, NumBSAnt))
            
    def Channel_square_PUE(model, idx_group, idx_square_UE): # without onoff
        
        # return sum((H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].real + RIS_received_real(model, idx_group, idx_square_UE, (idx_square_UE*NumBSAnt + idx_BS)))**2 +\
        #            (H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].imag + RIS_received_imag(model, idx_group, idx_square_UE, (idx_square_UE*NumBSAnt + idx_BS)))**2 for idx_BS in range(0, NumBSAnt))
        
        return sum((H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].real - RIS_received_real(model, idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 +\
                    (H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].imag - RIS_received_imag(model, idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 for idx_BS in range(0, NumBSAnt))
        
    
    def throughput_PUE(model):
        
        # return sum(RIS_square(model, 1, idx_SUE)*PUE_power[idx_pu] for idx_pu in range(NumPUE)) + RIS_square(model, 1, idx_SUE)*SUE_power[1 - idx_SUE]
        # return sum(RIS_square_PUE(model, 1, idx_SUE)*PUE_power[idx_pu] for idx_pu in range(NumPUE)) + RIS_square(model, 1, idx_SUE)*model.power[1 - idx_SUE]
        return sum((0.5*((log(1 + (( power_transform(model, idx_PUE_obj) * Channel_square(model, 0, idx_PUE_obj)) / sigma2)) / log(2) ) + \
                    (log(1 + ((power_transform(model, idx_PUE_obj) * Channel_square_PUE(model, 0, idx_PUE_obj)) / sigma2)) / log(2) )) ) for idx_PUE_obj in range(NumPUE))
    
    def throughput_SUE(model, L_spreadingfactor):
        
        # return sum(RIS_square_PUE(model, 0, idx_PUEI)*model.power[idx_su] for idx_su in range(NumSUE)) + RIS_square_PUE(model, 0, idx_PUEI)*PUE_power[1 - idx_PUEI]
        # return sum(RIS_square(model, 0, idx_PUEI)*model.power[idx_su] for idx_su in range(NumSUE)) + RIS_square_PUE(model, 0, idx_PUEI)*PUE_power[1 - idx_PUEI]
        return sum(( (1/L_spreadingfactor) * (log(1 + ((L_spreadingfactor * power_transform(model, (idx_SUE_obj//SUE_times)) * Channel_square(model, 1, idx_SUE_obj)) / sigma2)) / log(2) ) ) for idx_SUE_obj in range(NumSUE))
    
    def power_consumption(model, PT_static, RIS_static, SIC_dissapation, PUE_static, SUE_static):
        
        
        return sum((power_transform(model, idx_UE_obj)*mu + PUE_static + SIC_dissapation) for idx_UE_obj in range(NumPUE)) + \
            NumSUE * (SUE_static + SIC_dissapation) + PT_static + NumRISEle*RIS_static
    
    
    def phase_shift_discrete(model, idx_phase):
        
        # return (2*ma.pi*model.theta_discrete[idx_phase])/(2**NumUE)
        # return (2*ma.pi*model.theta_discrete[idx_phase])/(NumRISEle)
        
        return (2*ma.pi*model.theta_discrete[idx_phase])/(8)     # Fixed 8
        # return (2*ma.pi*Phase_shift_random[idx_phase])/(4)       # Random quantize
        # return Phase_shift_random[idx_phase]                       # Random phase shift
        
    def power_transform(model, idx_power):
        # print(idx_power)
        
        return pow(10, (model.power[idx_power]/10))/pow(10,3)
    
    
    
    def RIS_received_real_test(idx_realg, idx_risreal, phase_shift_test):
        
        # print("phase", phase_shift_test[idx_risreal])
        
        return sum((T_Conbime[idx_realg][idx_risreal][idx_ele].real*cos(phase_shift_test[idx_ele]) - \
                    T_Conbime[idx_realg][idx_risreal][idx_ele].imag*sin(phase_shift_test[idx_ele])) for idx_ele in range(0, NumRISEle))
            
    def RIS_received_imag_test(idx_imagg, idx_risimag, phase_shift_test):
        
        return sum((T_Conbime[idx_imagg][idx_risimag][idx_ele].real*sin(phase_shift_test[idx_ele]) + \
                    T_Conbime[idx_imagg][idx_risimag][idx_ele].imag*cos(phase_shift_test[idx_ele] ))  for idx_ele in range(0, NumRISEle))
            
    def Channel_square_test(idx_group, idx_square_UE, phase_shift_test): # wiith onoff
    
        return sum((H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].real +  RIS_received_real_test(idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 +\
                    (H_U2B_RicRe[idx_group][idx_square_UE][idx_BS].imag + RIS_received_imag_test(idx_group, (idx_square_UE*NumBSAnt + idx_BS)))**2 for idx_BS in range(0, NumBSAnt))
    
    
    
    
    # def constraints_01(model, idx_cons):
        
    #     # return t_b*Bandwidth*(log(1+((power_quantized(model, idx_cons) * RIS_square(model, idx_cons)) /sigma2)) / log(2)) >= LocalDataSize/T_e
    #     return sqrt(cos(model.theta[idx_cons])**2 + sin(model.theta[idx_cons])**2) <= 1
    
    def constraints_02(model, idx_pcons):
        
        # return t_b*Bandwidth*(log(1+((power_quantized(model, idx_cons) * RIS_square(model, idx_cons)) /sigma2)) / log(2)) >= LocalDataSize/T_e
        return model.power[idx_pcons] >= 0
    
    def constraints_03(model):
        
        return sum(power_transform(model, idx_pcons_max) for idx_pcons_max in range(NumPUE)) <= pow(10, (P_max/10))/pow(10,3)
    
    def constraints_04(model, idx_qos_p):
        
        # return t_b*Bandwidth*(log(1+((power_quantized(model, idx_cons) * RIS_square(model, idx_cons)) /sigma2)) / log(2)) >= LocalDataSize/T_e
        # return Bandwidth * (log(1+((PUE_power[idx_pqos] * RIS_square_PUE(model, 0, idx_pqos)) /(interference_PUE(model, idx_pqos) + sigma2))) / log(2)) >= Bandwidth * (log(sqrt(2))/log(2)) 
        return (0.5*((log(1 + ((power_transform(model, idx_qos_p) * Channel_square(model, 0, idx_qos_p)) / sigma2)) / log(2) ) + \
                    (log(1 + ((power_transform(model, idx_qos_p) * Channel_square_PUE(model, 0, idx_qos_p)) / sigma2)) / log(2) )) ) >= P_Rmin
    
    def constraints_05(model, idx_qos_s, L_spreadingfactor):
        
        # return t_b*Bandwidth*(log(1+((power_quantized(model, idx_cons) * RIS_square(model, idx_cons)) /sigma2)) / log(2)) >= LocalDataSize/T_e
        # return Bandwidth * (log(1+((PUE_power[idx_pqos] * RIS_square_PUE(model, 0, idx_pqos)) /(interference_PUE(model, idx_pqos) + sigma2))) / log(2)) >= Bandwidth * (log(sqrt(2))/log(2)) 
        return ( (1/L_spreadingfactor) * (log(1 + ((L_spreadingfactor * power_transform(model, (idx_qos_s//SUE_times)) * Channel_square(model, 1, idx_qos_s)) / sigma2)) / log(2) ) ) >= S_Rmin 
    
    
    
    model = ConcreteModel(name="Symbiotic_Radio_BB_test")
    
    model.power = Var([i_UE for i_UE in range(NumPUE)], bounds=(1,P_max), within=Integers, initialize = BB_power_ini)
    model.theta_discrete = Var([i for i in range(NumRISEle)], bounds=(-4, 4), within=Integers, initialize = BB_theta_ini)
    
    # model.xnf = Var([i_UE for i_UE in range(RISgroup)], bounds=(0,1), within=Binary, initialize = BB_onoff_ini)
    
    model.cons = ConstraintList()
    
    # for idx_cons1 in range(RISgroup * NumRISEle):
    #     model.cons.add(constraints_01(model, idx_cons1))
        
    for idx_cons2 in range(NumPUE):
        model.cons.add(constraints_02(model, idx_cons2))
    
    model.cons.add(constraints_03(model))
    
    for idx_cons4 in range(NumPUE):
        model.cons.add(constraints_04(model, idx_cons4))
        
    for idx_cons5 in range(NumSUE):
        model.cons.add(constraints_05(model, idx_cons5, L_spreadingfactor))
    
    model.obj = Objective(expr=obj_function(model, L_spreadingfactor, PT_static, RIS_static, SIC_dissapation, PUE_static, SUE_static), sense=maximize)  #-----------------setting objective function-------------------#

    # # solver_path = '/home/dddd/ris_gbd_solver/Bonmin-1.8.8/build/bin/bonmin'  #-----------------setting objective function-------------------#
    # solver_path = '/home/dddd/Solver_Couenne/Couenne-0.5.8/build/bin/couenne'
    solver_path = '/home/dddd/Solver_Couenne/Couenne-0.5.8/build/bin/bonmin'

    opt = SolverFactory('bonmin', executable=solver_path)
    opt.options['bonmin.algorithm'] = 'B-BB'
    # opt.options['bonmin.algorithm'] = 'B-OA'
    # # opt.options['nlp_scaling_method'] = 'gradient-based'
    # # opt.options['nlp_scaling_max_gradient'] = int(1e3)
    # opt.options['mu_strategy'] = 'adaptive'
    

    results = opt.solve(model, tee=False)
    # results = opt.solve(model, tee=True)
    # # results.write()
    model.cons.display()
    #------------------------------------------------------------------------------------------------------------------------------#
    # model_theta_BB_0 = np.zeros([NumRISEle, 1], dtype= float)
    # model_theta_BB_1 = np.zeros([NumRISEle, 1], dtype= float)
    
    model_theta_BB = np.array(list(model.theta_discrete.get_values().values()))
    # for idx_t in range(NumRISEle):
    #     model_theta_BB_0[idx_t] = model_theta_BB[idx_t]
    #     model_theta_BB_1[idx_t] = model_theta_BB[idx_t + NumRISEle]
    
    
    model_power_BB = np.array(list(model.power.get_values().values()))

    
    for idx_t in range(NumPUE):
        model_power_BB[idx_t] = pow(10, (model_power_BB[idx_t]/10))/pow(10,3)
    
    # model_onoff_BB = np.array(list(model.xnf.get_values().values()))
    # # model_onoff_BB = np.array([0,1])
    
    Symbiotic_BB_status = results.solver.termination_condition
    Symbiotic_BB_objective = model.obj()
    
    # # # print("power BB : ", model_power_BB)
    # # # print("theta BB : ", model_theta_BB)
    # # # print("PS : ", np.diag(np.exp(1j*(2*ma.pi*model_theta_BB/(2**NumUE)))))
    
    
    # # # print("Diag PS : ", np.exp(1j*Phase_shift_test))
    # # # print("Diag PS : ", np.eye(NumRISEle) * np.exp(1j*Phase_shift_test)) 
    
    # RIS_matrix_multiply_P = S_U2R_Ric @ (np.eye(NumRISEle) * np.exp(1j*Phase_shift_test_0)) @ H_R2B_Ric
    # RIS_matrix_multiply_S = H_U2R_Ric @ (np.eye(NumRISEle) * np.exp(1j*Phase_shift_test_1)) @ H_R2B_Ric
    RIS_matrix_multiply_P = H_U2R_Ric @ (np.diag(np.exp(1j*(2*ma.pi*model_theta_BB/(8))))) @ H_R2B_Ric
    RIS_matrix_multiply_S = S_U2R_Ric @ (np.diag(np.exp(1j*(2*ma.pi*model_theta_BB/(8))))) @ H_R2B_Ric
    
    # print("RIS matrix P : \n", RIS_matrix_multiply_P)
    # print("RIS matrix S : \n", RIS_matrix_multiply_S)
    
    # model_theta_BB_test =  (2*ma.pi*model_theta_BB/(8))
    
    # RIS_link_realtest = []
    # RIS_link_imagtest = []
    
    # flag_solver = 0
    
    # for idx_r in range(NumBSAnt):
        
    #     RIS_link_realtest.append(RIS_received_real_test(0, idx_r, model_theta_BB_test))
    
    # for idx_i in range(NumBSAnt):
        
    #     RIS_link_imagtest.append(RIS_received_imag_test(1, idx_i, model_theta_BB_test))
        
    # print("RIS solver P : \n", RIS_matrix_multiply_P)
    # print("RIS solver S : \n", RIS_matrix_multiply_S)
    
    # RIS_matrix_multiply_S0 = S_U2R_Ric[0,:] @ (np.eye(NumRISEle) * np.exp(1j*(model_theta_BB))) @ H_R2B_Ric
    # RIS_matrix_multiply_S1 = S_U2R_Ric[1,:] @ (np.eye(NumRISEle) * np.exp(1j*(model_theta_BB))) @ H_R2B_Ric
    
    # RIS_matrix_multiply_P0 = H_U2R_Ric[0,:] @ (np.eye(NumRISEle) * np.exp(1j*(model_theta_BB))) @ H_R2B_Ric
    # RIS_matrix_multiply_P1 = H_U2R_Ric[1,:] @ (np.eye(NumRISEle) * np.exp(1j*(model_theta_BB))) @ H_R2B_Ric
    
    
    
    # Total_link_S0 = H_U2B_RicRe[1][0,:] + model_onoff_BB[0] * RIS_matrix_multiply_S0
    # Total_link_S1 = H_U2B_RicRe[1][1,:] + model_onoff_BB[1] * RIS_matrix_multiply_S1
    
    # Total_link_S0_without = H_U2B_RicRe[1][0,:] + RIS_matrix_multiply_S0
    # Total_link_S1_without = H_U2B_RicRe[1][1,:] + RIS_matrix_multiply_S1
    
    # Total_link_P0 = H_U2B_RicRe[0][0,:] + model_onoff_BB[0] *  RIS_matrix_multiply_P0
    # Total_link_P1 = H_U2B_RicRe[0][1,:] + model_onoff_BB[1] * RIS_matrix_multiply_P1
    
    # Total_link_P0_without = H_U2B_RicRe[0][0,:] + RIS_matrix_multiply_P0
    # Total_link_P1_without = H_U2B_RicRe[0][1,:] + RIS_matrix_multiply_P1
    
    Total_link_P = H_U2B_Ric + RIS_matrix_multiply_P
    Total_link_P_minus = H_U2B_Ric - RIS_matrix_multiply_P
    Total_link_S = S_U2B_Ric + RIS_matrix_multiply_S
    
    Diag_square_P = np.diag(Total_link_P @ Total_link_P.conj().T)
    Diag_square_P_minus = np.diag(Total_link_P_minus @ Total_link_P_minus.conj().T)
    Diag_square_S = np.diag(Total_link_S @ Total_link_S.conj().T)
        
    # Diag_square_S0 = Total_link_S0 @ Total_link_S0.conj().T
    # Diag_square_S1 = Total_link_S1 @ Total_link_S1.conj().T
    # Diag_square_S = np.array([Diag_square_S0, Diag_square_S1])
    
    # Diag_square_S0_without = Total_link_S0_without @ Total_link_S0_without.conj().T
    # Diag_square_S1_without = Total_link_S1_without @ Total_link_S1_without.conj().T
    # Diag_square_S_without = np.array([Diag_square_S0_without, Diag_square_S1_without])
    
    # Diag_square_P0 = Total_link_P0 @ Total_link_P0.conj().T
    # Diag_square_P1 = Total_link_P1 @ Total_link_P1.conj().T
    # Diag_square_P = np.array([Diag_square_P0, Diag_square_P1])
    
    # Diag_square_P0_without = Total_link_P0_without @ Total_link_P0_without.conj().T
    # Diag_square_P1_without = Total_link_P1_without @ Total_link_P1_without.conj().T
    # Diag_square_P_without = np.array([Diag_square_P0_without, Diag_square_P1_without])
    
    # Throughput_P_sum = 0
    # Throughput_S_sum = 0
    # Total_power = 0
    
    # Throughput_P_sum = complex(Throughput_P_sum)
    # Throughput_S_sum = complex(Throughput_S_sum)
    # Total_power = complex(Total_power)
    
    Throughput_P_sum = sum( 0.5 * ( ma.log2(1 + ( model_power_BB[idx_sum_p] * Diag_square_P[idx_sum_p] / sigma2) ) +  \
                                   ma.log2(1 + (model_power_BB[idx_sum_p] * Diag_square_P_minus[idx_sum_p] / sigma2) ) ) for idx_sum_p in range(NumPUE) )
        
    Throughput_S_sum = sum( (1/L_spreadingfactor) * ( ma.log2(1 +  (L_spreadingfactor * model_power_BB[(idx_sum_s//SUE_times)] * Diag_square_S[idx_sum_s] / sigma2) ) ) 
                           for idx_sum_s in range(NumSUE))
    
    Total_power = sum( model_power_BB[idx_power_p]*mu + PUE_static + SIC_dissapation for idx_power_p in range(NumPUE) ) + \
                    NumSUE * (SUE_static + SIC_dissapation) + PT_static + NumRISEle*RIS_static
    
    # Throughput_P_sum = Throughput_P_sum.real
    # Throughput_S_sum = Throughput_S_sum.real
    # Total_power = Total_power.real
    
    # # Combine_channel_square = np.zeros(NumUE)
    # # flag_square = 0
    
    # # for idx_v in range(Group_UE):
    # #     Combine_channel_square[flag_square] = RIS_square(model, 1, idx_v)
    # #     flag_square += 1
    
    # print("Channel square : ", Diag_square_S)
    # # print("Solver square : ",Combine_channel_square )
    
    # # Solver_obj = obj_function(model)
    
    Obj_veri = (Throughput_P_sum + Throughput_S_sum) / Total_power
    # for idx_j in range(NumSUE):
        
    #     Interf = sum(PUE_power[idx_p]*Diag_square_S_without[idx_j] for idx_p in range(NumPUE)) + model_power_BB[1 - idx_j]*Diag_square_S[idx_j]
    #     Obj_veri += Bandwidth*ma.log2( 1 + (model_power_BB[idx_j]*Diag_square_S[idx_j]/(Interf + sigma2))) / (mu*model_power_BB[idx_j] + P_k) 
    
    # Interference_PUE = []
    # Interf_P = 0
    # for idx_k in range(NumPUE):
        
    #     Interf_P = sum(Diag_square_P[idx_k]*model_power_BB[idx_pt] for idx_pt in range(NumSUE)) + Diag_square_P_without[idx_k]*PUE_power[1 - idx_k]
    #     # Interference_PUE.append(Bandwidth*ma.log2(1+((PUE_power[idx_k] * Diag_square_P_without[idx_k]) /(Interf_P + sigma2))))
    #     Interference_PUE.append(((PUE_power[idx_k] * Diag_square_P_without[idx_k]) /(Interf_P + sigma2)))
    # # Obj_veri = complex(0)
    # # for idx_result in range(NumUE):
        
    # #     Obj_veri += (T_e*t_b*Bandwidth*ma.log2(1+((model_power_BB[idx_result]*poewr_times)*Diag_square[idx_result]/sigma2)))/(T_e*t_b*(model_power_BB[idx_result]* poewr_times))
    
    # # # print("Received real : ", Received_channel_real)
    # # # print("Received imag : ", Received_channel_imag)
    
    # # # print("Received RIS matrix : ", RIS_link_real + 1j*RIS_link_imag)
    
    # # # print("Received matrix : ", Received_channel_real + 1j*Received_channel_imag)
    # # # print("Channel gain", channel_square)
    
    # print("power BB :\n", model_power_BB)
    # print("theta BB :\n", model_theta_BB)
    # print("onoff BB :\n", model_onoff_BB)
    
    # print("DQN BB obj : ", DQN_BB_objective)
    
    sum_power = 0
    sum_power = sum(model_power_BB)
    # for idx_cal in range(NumPUE):
    #     sum_power += pow(10, (model_power_BB[idx_cal]/10))/pow(10,3)
    
    
    print("DQN BB status: ", Symbiotic_BB_status)
    print("Power consume : ", sum_power, ":W")
    print("Solver Obj : ", Symbiotic_BB_objective)
    # print("*************************************************************************")
    
    # # # print("RIS matrix link : ", RIS_matrx_multiply)
    
    # # # print("Total link 2 : ",  Total_link_2)
    # # # print("Square martix : ", Diag_square)
    print("Throughput_P : ", Throughput_P_sum, "\nThroughput_S : ", Throughput_S_sum, "\nTotal_power : ", Total_power)
    print("obj veri : ", Obj_veri)
    # print("Interference PUE: ", Interference_PUE)
    Symbiotic_BB_time = results.solver.time
    
    
    # return 0
    return Symbiotic_BB_status, Symbiotic_BB_objective, model_power_BB, model_theta_BB, Symbiotic_BB_time
