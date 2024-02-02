"""
Multi-user MIMO, environment
"""

from __future__ import division
import numpy as np
import math 


seed = 0
np.random.seed(seed)

class MultiUE_ENV:
    def __init__(self, NumBSAnt, NumRISEle, NumUE, NumPUE, NumSUE):
        self.NumBSAnt = NumBSAnt    # M
        self.NumRISEle = NumRISEle  # L
        self.NumUE = NumUE          # K
        self.NumPUE = NumPUE 
        self.NumSUE = NumSUE 
        # np.random.seed(seed)
        
        # Bandwidth=1 #MHz
        # Noise_o=-104 #dBm  -174dBm/Hz
        # Noise=pow(10, (Noise_o/10))/pow(10,3) #watts
        
        
        # #Area: Square
        # lengA = -500 
        # lengB = 60 # m
        
        # #power limits
        
        # mu=5/4                                   #---------power amplifier efficiency ^ -1---------#
        
        # P_max_o=50 #dBm
        # P_max=pow(10, (P_max_o/10))/pow(10,3)    #------------Maximum transmitted power------------#
        # P_k_o=10  #dBm
        # P_k=pow(10, (P_k_o/10))/pow(10,3)        #----------User static power comsumption----------#
        
        
        
        # #------------BS location-----------------#

        # BS_loc=np.array([lengA/2,lengA/2, 10])
        


        # #-----------RIS location-----------------#

        # RISloc=np.zeros([RIS_Lnum, 3])

        

        # for i in range(0,RIS_Lnum):   # i = l_RIS, i從0開始

        #     RISloc[i] = [0, 0, 20]
            
            

        # #---------------User location-----------------#

        # User_loc = np.zeros([Num_User, 3])
        
        # #--------------Ramdom or Fixed user location------------#
        
        # if seed != 15:
            
        #     User_loc[0] =[lengB*np.random.rand(1), lengB*np.random.rand(1), 1]
        # else:
        #     User_loc[0] =[20, 20, 1]
       

        # for i in range(0,Num_User):   #i = userloc
        #         User_loc[i] = User_loc[0]
                
        # dismin = 10
        
        # PL_0 = ma.pow(10, (-30/10))

    # JSAC absolute phase shift (DFT vectors)
    # def DFT_matrix(self, N_point):  # N_point: 決定有多少種相位，if N_point=4，則從 0*pi/2 ~ 3*pi/2
    #     n, m = np.meshgrid(np.arange(N_point), np.arange(N_point))
    #     omega = np.exp(-2 * math.pi * 1j / N_point)
    #     W = np.power( omega, n * m ) 
    #     return W
    
    # # 用於生成 Steering Vector
    # def SubSteeringVec(self, Angle, NumAnt):
    #     SSV = np.exp(1j * Angle * math.pi * np.arange(0, NumAnt, 1))
    #     SSV = SSV.reshape(-1, 1)    # 行的元素數量固定為1，並自動生成列 (-1)
    #     return SSV
    
    # LoS channel response, which is position dependent 
    # def ChannelResponse(self, Pos_A, Pos_B, ArrayShape_A, ArrayShape_B):   
    #     Dis_AB = np.linalg.norm(Pos_A - Pos_B)                                  ## distance between AB (2-norm: 平方合開根號)
    #     DirVec_AB = (Pos_A - Pos_B) / Dis_AB                                    ## normalized direction vector (每個位置上放的是向量投影到 xyz 軸與向量夾角的 cosine values)
    #     SteeringVectorA = np.kron(self.SubSteeringVec(DirVec_AB[0], ArrayShape_A[0]), self.SubSteeringVec(DirVec_AB[1], ArrayShape_A[1]))
    #     SteeringVectorA = np.kron(SteeringVectorA, self.SubSteeringVec(DirVec_AB[2], ArrayShape_A[2]))  # 根據天線所放的軸，計算它的 steering vectoor，其他部分只會是1，所以做 kron 答案不影響
    #     SteeringVectorB = np.kron(self.SubSteeringVec(DirVec_AB[0], ArrayShape_B[0]), self.SubSteeringVec(DirVec_AB[1], ArrayShape_B[1]))
    #     SteeringVectorB = np.kron(SteeringVectorB, self.SubSteeringVec(DirVec_AB[2], ArrayShape_B[2]))  # 根據天線所放的軸，計算它的 steering vectoor，其他部分只會是1，所以做 kron 答案不影響
    #     SteeringVectorB_H = np.matrix.getH(SteeringVectorB)
    #     H_LoS_matrix = SteeringVectorA @ SteeringVectorB_H                      # size_A x 1 的矩陣 @ 1 x size_B 的矩陣
    #     return H_LoS_matrix

    # Generate LOS channel for Rician fading
    # def H_GenLoS(self, Pos_BS, Pos_RIS, Pos_UE, ArrayShape_BS, ArrayShape_RIS, ArrayShape_UE):   # for multi-user channel (2)
    #     H_R2B_LoS = self.ChannelResponse(Pos_BS, Pos_RIS, ArrayShape_BS, ArrayShape_RIS)
    #     H_U2B_LoS = np.zeros((self.NumBSAnt, self.NumUE), dtype = complex)
    #     H_U2R_LoS = np.zeros((self.NumRISEle, self.NumUE), dtype = complex) 
    #     # H = [h1, h2, · · · , hK]，一個UE對應一行
    #     for i in range(self.NumUE):
    #         h_U2B_LoS = self.ChannelResponse(Pos_BS, Pos_UE[i], ArrayShape_BS, ArrayShape_UE)    # NumBSAnt x 1
    #         H_U2B_LoS[:, i] = h_U2B_LoS.reshape(-1)                                              # .reshape(-1): 固定一列，自動生成行。這步驟是將向量放入該矩陣的第 i 行
    #         h_U2R_LoS = self.ChannelResponse(Pos_RIS, Pos_UE[i], ArrayShape_RIS, ArrayShape_UE)  # NumRISEle x 1   
    #         H_U2R_LoS[:, i] = h_U2R_LoS.reshape(-1)
    #     return H_U2B_LoS, H_R2B_LoS, H_U2R_LoS
    
    # Generate Rayleigh channel fading or NLOS channel for Rician fading
    def H_GenNLoS(self, ):
        
        P_U2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumPUE, self.NumBSAnt)) + 1j * np.random.normal(0, 1, size=(self.NumPUE, self.NumBSAnt)))
        H_R2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumRISEle, self.NumBSAnt)) + 1j * np.random.normal(0, 1, size=(self.NumRISEle, self.NumBSAnt)))
        P_U2R_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumPUE, self.NumRISEle)) + 1j * np.random.normal(0, 1, size=(self.NumPUE, self.NumRISEle)))
        
        S_U2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumSUE, self.NumBSAnt)) + 1j * np.random.normal(0, 1, size=(self.NumSUE, self.NumBSAnt)))
        # S_R2B_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumRISEle, self.NumBSAnt)) + 1j * np.random.normal(0, 1, size=(self.NumRISEle, self.NumBSAnt)))
        S_U2R_NLoS = 1 / math.sqrt(2) * (np.random.normal(0, 1, size=(self.NumSUE, self.NumRISEle)) + 1j * np.random.normal(0, 1, size=(self.NumSUE, self.NumRISEle)))
        
        return P_U2B_NLoS, H_R2B_NLoS, P_U2R_NLoS, S_U2B_NLoS, S_U2R_NLoS
    
    # Generate Large-scale path loss
    def H_GenPL(self, Pos_BS, Pos_RIS, Pos_UE):   
        # Large-scale pass loss (參數參考 Energy-Efficient Federated Learning With Intelligent Reflecting Surface)
        PL_0 = 10**(-30/10);                                # dB the channel gain at the reference distance
        d_R2B = np.linalg.norm(Pos_RIS - Pos_BS)            # distance from the RIS to BS  
        pathloss_R2B = math.sqrt(PL_0 * (d_R2B)**(-2.2));   # Large-scale pass loss from RIS to BS (α is the path loss exponent, α = 2.2)                                                                
        P_d_U2B = np.zeros(self.NumPUE)  
        P_d_U2R = np.zeros(self.NumPUE)
        S_d_U2B = np.zeros(self.NumSUE)  
        S_d_U2R = np.zeros(self.NumSUE) 
        
        P_pathloss_U2B = np.zeros(self.NumPUE)  
        P_pathloss_U2R = np.zeros(self.NumPUE)
        S_pathloss_U2B = np.zeros(self.NumSUE)  
        S_pathloss_U2R = np.zeros(self.NumSUE)
        for k in range(self.NumPUE):
            P_d_U2B[k] = np.linalg.norm(Pos_UE[k] - Pos_BS)      # distance from the user k to the BS  
            P_d_U2R[k] = np.linalg.norm(Pos_UE[k] - Pos_RIS)      # distance from the user k to the RIS  
            P_pathloss_U2B[k] = math.sqrt(PL_0 * (P_d_U2B[k])**(-3.5))   # Large-scale pass loss from user k to BS (α is the path loss exponent, α = 4)
            P_pathloss_U2R[k] = math.sqrt(PL_0 * (P_d_U2R[k])**(-2.2))   # Large-scale pass loss from user k to RIS (α is the path loss exponent, α = 2)
            
        for k in range(self.NumPUE, self.NumUE):
            S_d_U2B[k - self.NumPUE] = np.linalg.norm(Pos_UE[k] - Pos_BS)      # distance from the user k to the BS  
            S_d_U2R[k - self.NumPUE] = np.linalg.norm(Pos_UE[k] - Pos_RIS)      # distance from the user k to the RIS  
            S_pathloss_U2B[k - self.NumPUE] = math.sqrt(PL_0 * (S_d_U2B[k - self.NumPUE])**(-3.5))   # Large-scale pass loss from user k to BS (α is the path loss exponent, α = 4)
            S_pathloss_U2R[k- self.NumPUE] = math.sqrt(PL_0 * (S_d_U2R[k - self.NumPUE])**(-2.2))   # Large-scale pass loss from user k to RIS (α is the path loss exponent, α = 2)
        
        return P_pathloss_U2B, pathloss_R2B, P_pathloss_U2R, S_pathloss_U2B, S_pathloss_U2R
    
    # # The channel include large-scale fading and small-scale fading.
    # def H_RicianOverall(self, K_U2B, K_R2B, K_U2R, H_U2B_LoS, H_R2B_LoS, H_U2R_LoS, H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS, pathloss_U2B, pathloss_R2B, pathloss_U2R):
    #     H_R2B_temp = (math.sqrt(1 / (K_R2B + 1)) * H_R2B_NLoS + math.sqrt(K_R2B / (K_R2B + 1)) * H_R2B_LoS)
    #     H_U2B_temp = (math.sqrt(1 / (K_U2B + 1)) * H_U2B_NLoS + math.sqrt(K_U2B / (K_U2B + 1)) * H_U2B_LoS) 
    #     H_U2R_temp = (math.sqrt(1 / (K_U2R + 1)) * H_U2R_NLoS + math.sqrt(K_U2R / (K_U2R + 1)) * H_U2R_LoS)
    #     H_R2B_Ric = pathloss_R2B * H_R2B_temp
    #     H_U2B_Ric = np.zeros((self.NumBSAnt, self.NumUE), dtype = complex)
    #     H_U2R_Ric = np.zeros((self.NumRISEle, self.NumUE), dtype = complex) 
    #     for i in range(self.NumBSAnt):
    #         for k in range(self.NumUE):
    #             H_U2B_Ric[i][k] = pathloss_U2B[k] * H_U2B_temp[i][k]
    #     for i in range(self.NumRISEle):
    #         for k in range(self.NumUE):
    #             H_U2R_Ric[i][k] = pathloss_U2R[k] * H_U2R_temp[i][k]
    #     return H_U2B_Ric, H_R2B_Ric, H_U2R_Ric
        
    # The channel include large-scale fading and small-scale fading.
    def H_RayleighOverall(self, P_U2B_NLoS, H_R2B_NLoS, P_U2R_NLoS, S_U2B_NLoS, S_U2R_NLoS,
                          P_pathloss_U2B, pathloss_R2B, P_pathloss_U2R, S_pathloss_U2B, S_pathloss_U2R):
        H_B2R_Ray = pathloss_R2B * H_R2B_NLoS
        P_B2U_Ray = np.zeros((self.NumPUE, self.NumBSAnt), dtype = complex)
        P_R2U_Ray = np.zeros((self.NumPUE, self.NumRISEle), dtype = complex)
        S_B2U_Ray = np.zeros((self.NumSUE, self.NumBSAnt), dtype = complex)
        S_R2U_Ray = np.zeros((self.NumSUE, self.NumRISEle), dtype = complex)
        
        for i in range(self.NumBSAnt):
            for k in range(self.NumPUE):
                P_B2U_Ray[k][i] = P_pathloss_U2B[k] * P_U2B_NLoS[k][i]
        for i in range(self.NumRISEle):
            for k in range(self.NumPUE):
                P_R2U_Ray[k][i] = P_pathloss_U2R[k] * P_U2R_NLoS[k][i]
                
        for i in range(self.NumBSAnt):
            for k in range(self.NumSUE):
                S_B2U_Ray[k][i] = S_pathloss_U2B[k] * S_U2B_NLoS[k][i]
        for i in range(self.NumRISEle):
            for k in range(self.NumSUE):
                S_R2U_Ray[k][i] = S_pathloss_U2R[k] * S_U2R_NLoS[k][i]
                
        return P_B2U_Ray, H_B2R_Ray, P_R2U_Ray, S_B2U_Ray, S_R2U_Ray
    
    # # Effective kth-device-BS combined channel
    # def H_Comb(self, H_U2B, H_R2B, H_U2R, RefVector): 
    #     RefPattern_matrix = np.diag(RefVector)  
    #     # print('RefPattern_matrix: \n', RefPattern_matrix)
    #     # H = H_U2B + 1*np.linalg.multi_dot([H_R2B, RefPattern_matrix, H_U2R])                # np.linalg.multi_dot: 计算两个或多个矩阵的乘积
    #     H = H_U2B + H_R2B @ RefPattern_matrix @ H_U2R                                           # @: 矩陣乘法   # *:對應元素相乘    
    #     return H

    # # Throughput of each user (SNR)
    # def SNR_Throughput(self, H_Ric, BW, LocalDataSize, sigma2, Power_UE): 
    #     # 分子 Rician 分母 Rayleigh (分子分母不同通道)
    #     H_Ric_gain2 = abs(np.conj(H_Ric.T) @ H_Ric) # 取決對值讓複數變長度
    #     H_Ric_gain2 = np.diag(H_Ric_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
    #     # print('Power_UE: ', Power_UE)
    #     SigPower = Power_UE * H_Ric_gain2           # Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方在乘上所對應的 UE 的發射功率
    #     SNR = SigPower / (0 + sigma2)
    #     # print('SNR: \n', SNR)
    #     Rate = np.log2(1 + SNR)
    #     Rate = Rate.reshape(1, self.NumUE)      # 回傳後才可以放到STATE
    #     Throughput = BW * np.log2(1 + SNR)

    #     # 計算 EE ，不發射功率的 EE 為 0
    #     # Power_UE = Power_UE.reshape(1, self.NumUE)       # 調整後才可以被下面的拿去用    
    #     EE = np.ones(self.NumUE)
    #     EE[Power_UE == 0] = 0                        # 把功率為0的對應位置的EE設為0
    #     EE[Power_UE != 0] = Throughput[Power_UE != 0] / Power_UE[Power_UE != 0]     # 計算功率不為0的對應位置的EE
    #     EE = EE.reshape(1, self.NumUE)          # 回傳後才可以放到STATE
    #     return SNR, Rate, Throughput, EE
    
    # # --------------------------------------------------------------------------
    # # --------------------------------------------------------------------------
    # # Throughput of each user (SINR)
    # def SINR_Throughput(self, H_Ric, H_Ray, BW, sigma2, Power_UE): 
    #     # 分子 Rician 分母 Rayleigh (分子分母不同通道)
    #     H_Ric_gain2 = abs(np.conj(H_Ric.T) @ H_Ric) # 取決對值讓複數變長度
    #     H_Ric_gain2 = np.diag(H_Ric_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
    #     H_Ray_gain2 = abs(np.conj(H_Ray.T) @ H_Ray)
    #     H_Ray_gain2 = np.diag(H_Ray_gain2)          # 從 KxK 的矩陣中取出 Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方
    #     SigPower = Power_UE * H_Ric_gain2           # Kx1 的向量，分別代表每個 UE 對應的 Channel gain 的平方在乘上所對應的 UE 的發射功率
    #     # SigPower = Power_UE * H_Ray_gain2
    #     Power_Channel_gain = Power_UE * H_Ray_gain2
    #     Power_Channel_gain_sum = Power_Channel_gain.sum()
    #     IntfPower = (Power_Channel_gain_sum - Power_Channel_gain)    
    #     SINR = SigPower / (IntfPower + sigma2)
    #     print('SINR: \n', SINR)
    #     Rate = np.log2(1 + SINR)
    #     Throughput = BW * np.log2(1 + SINR)
    #     return SINR, Rate, Throughput
    # # --------------------------------------------------------------------------
    # # --------------------------------------------------------------------------
    




        