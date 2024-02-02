#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:43:39 2023

@author: dddd
"""

import numpy as np
import math

class opt_function:
    
    def __init__(self, NumBSAnt, NumRISEle, NumUE, sigma2):
        
        self.NumBSAnt = NumBSAnt    # M
        self.NumRISEle = NumRISEle  # L
        self.NumUE = NumUE          # K
        # self.Bandwidth = Bandwidth
        self.sigma2 = sigma2
        # self.Gene_length = NumRISEle + 2*NumUE
        
    #objective function
    def obj_func(self, throughput_primary, throughput_secondary, P_consume):
        
        score = (throughput_primary + throughput_secondary) / P_consume
        
        
        return score
    
    #primary user throughput
    def primary_throughput(self, primary_power, Total_primary_channel, Total_minus_channel):
        
        throughput_primary = 0
        throughput_primary_UE = np.zeros(int(self.NumUE/2))
        channel_gain_primary = np.diag(Total_primary_channel @ Total_primary_channel.conj().T)
        channel_gain_minus = np.diag(Total_minus_channel @ Total_minus_channel.conj().T)
        
        for i in range( int(self.NumUE/2) ):
            
            throughput_primary_UE[i] = 0.5*( math.log2( 1 + (primary_power[i] * channel_gain_primary[i]) /self.sigma2) + \
                                    math.log2( 1 + (primary_power[i] * channel_gain_minus[i]) /self.sigma2))
            throughput_primary += throughput_primary_UE[i]
        
        return throughput_primary, throughput_primary_UE
    
    
    #secondary user throughput
    def secondary_throughput(self, primary_power, Total_secondary_channel, L_spreadingfactor):
        
        throughput_secondary = 0
        throughput_secondary_UE = np.zeros(int(self.NumUE/2))
        channel_gain_secondary = np.diag(Total_secondary_channel @ Total_secondary_channel.conj().T)
        
        for i in range( int(self.NumUE/2) ):
            
            throughput_secondary_UE[i] += (1/L_spreadingfactor) * \
                math.log2( 1 + (L_spreadingfactor * primary_power[i] * channel_gain_secondary[i]) /self.sigma2)
            
            throughput_secondary += throughput_secondary_UE[i]
        
        return throughput_secondary, throughput_secondary_UE
    
    
    #Total power consumption
    def power_consumption(self, primary_power, PTx_static, mu, SIC_dispation, RIS_static, PUE_static, SUE_static):
        
        Total_power = 0
        Total_power += mu * sum(primary_power[idx] for idx in range(int(self.NumUE/2)))
        Total_power += PTx_static
        Total_power += self.NumRISEle * RIS_static
        Total_power += (int(self.NumUE/2) * PUE_static) + (int(self.NumUE/2) * SUE_static) + (self.NumUE * SIC_dispation)
        
        return Total_power

    
    #Whloe Channel with RIS symbol = 1 
    def Total_channel_conbime(self, phase_shift, B2R, R2U, B2U):
        
        channel_Total = B2U + R2U @ phase_shift @ B2R
        
        return channel_Total
    
    #Whloe Channel with RIS symbol = -1 
    def Total_channel_minus(self, phase_shift, B2R, R2U, B2U):
        
        channel_Total_minus = B2U - R2U @ phase_shift @ B2R
        
        
        return channel_Total_minus