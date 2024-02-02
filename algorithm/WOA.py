#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:51:58 2023

@author: dddd
"""

import numpy as np
import math
import random
import timeit

import function_compute as FUNC

class Whale:
    
    def __init__(self, NumBSAnt, NumRISEle, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, P_Rmin, S_Rmin,
                 P_max, PTx_static, mu, SIC_dispation, RIS_static, PUE_static, SUE_static, B2R, P_B2U, P_R2U, S_B2U, S_R2U, whale_a, whale_b, whale_a_step):
        self.NumBSAnt = NumBSAnt    # M
        self.NumRISEle = NumRISEle  # L
        self.NumUE = NumUE          # K
        self.NumPUE = NumPUE 
        self.NumSUE = NumSUE
        self.sigma2 = sigma2
        self.L_spreadingfactor = L_spreadingfactor
        self.P_Rmin = P_Rmin
        self.S_Rmin = S_Rmin
        self.P_max = P_max
        self.population_size = population_size
        self.RIS_NumPhase = 2**RIS_bits
        self.mutate_rate = mutate_rate
        self.population_group = int(1/5*population_size)
        self.gene_length = self.NumRISEle + self.NumPUE
        self.whale_a = whale_a
        self.whale_a_step = whale_a_step
        self.whale_b = whale_b
        
        self.mu = mu
        self.PTx_static = PTx_static
        self.SIC_dispation = SIC_dispation
        self.RIS_static = RIS_static
        self.PUE_static = PUE_static
        self.SUE_static = SUE_static
        
        self.B2R = B2R
        self.P_B2U = P_B2U
        self.P_R2U = P_R2U
        self.S_B2U = S_B2U
        self.S_R2U = S_R2U
        # self.Gene_length = NumRISEle + 2*NumUE
        
        self.Funcall = FUNC.opt_function(self.NumBSAnt, self.NumRISEle, self.NumUE, self.sigma2)
    
    # def whale_rank_sol(self):
        
        
    #     return
    def Phase_shift_decode(self, phase_gene):
        
        Phase_shift_real = np.exp(1j*(2*math.pi*phase_gene/(8)))
        
        return Phase_shift_real
    
    
    
    def Power_decode(self, power_gene):
        
        Power_real = np.zeros(self.NumPUE)
        for idx_pow in range(self.NumPUE):
            Power_real[idx_pow] = pow(10, (power_gene[idx_pow]/10))/pow(10,3)
        
        return Power_real
    
    def whale_generation_pop(self):
        
        # gene_length = self.NumRISEle + self.NumPUE
        phase_range = int(self.RIS_NumPhase / 2)
        # print("PH :", phase_range, "\n Num phase : ", self.RIS_NumPhase)
        population = np.zeros([int(2*self.population_group), self.gene_length], dtype=np.float32)
        
        for idx_pop in range(int(2*self.population_group)):
            for idx_len in range(self.gene_length):
                
                if(idx_len >= self.NumRISEle):
                    population[idx_pop][idx_len] = random.randint(15, 30)
                
                else:
                    population[idx_pop][idx_len] = random.randint(-phase_range, phase_range)
                    
        return population
    
    def whale_fitness(self, individual_gene):
        
        score = 0
        
        
        phase_gene = individual_gene[0:self.NumRISEle]
        power_gene = individual_gene[self.NumRISEle:]
        
        phase_decode = self.Phase_shift_decode(phase_gene)
        power_decode = self.Power_decode(power_gene)
        
        if(sum(power_decode) > pow(10, (self.P_max/10))/pow(10,3)):
            score += -5
        
        Primary_channel = self.Funcall.Total_channel_conbime(np.diag(phase_decode), self.B2R, self.P_R2U, self.P_B2U)
        Secondary_channel = self.Funcall.Total_channel_conbime(np.diag(phase_decode), self.B2R, self.S_R2U, self.S_B2U)
        Primary_channel_minus = self.Funcall.Total_channel_minus(np.diag(phase_decode), self.B2R, self.P_R2U, self.P_B2U)
        
        Primary_throughput, Primary_throughput_UE = self.Funcall.primary_throughput(power_decode, Primary_channel, Primary_channel_minus)
        Secondary_throughput, Secondary_throughput_UE = self.Funcall.secondary_throughput(power_decode, Secondary_channel, self.L_spreadingfactor)
        Power_consumption = self.Funcall.power_consumption(power_decode, self.PTx_static, self.mu, self.SIC_dispation, self.RIS_static, self.PUE_static, self.SUE_static)
        
        for idx_fp in range(self.NumPUE):
            if( Primary_throughput_UE[idx_fp] < self.P_Rmin):
                score += (-10)*(self.P_Rmin - Primary_throughput_UE[idx_fp])
                
        for idx_fs in range(self.NumSUE):
            if( Secondary_throughput_UE[idx_fs] < self.S_Rmin):
                score += (-100)*(self.S_Rmin - Secondary_throughput_UE[idx_fs])
        
        
        temp_obj = self.Funcall.obj_func(Primary_throughput, Secondary_throughput, Power_consumption)
        score += temp_obj
        
        return score#,temp_obj
    
    def A_compute(self):
        
        rand_vecA = np.random.rand(self.gene_length)
        
        return self.whale_a *(2*rand_vecA - 1)
    
    
    def C_compute(self):
        
        return 2.0 * np.random.rand(self.gene_length)
    
    
    def D_compute(self, current_sol, best_rand_sol):
        
        C = self.C_compute()
        
        return np.linalg.norm(np.multiply(C, best_rand_sol) - current_sol) 
    
    
    def Next_iter_position(self, current_sol, best_rand_sol, A_vec):
        
        D = self.D_compute(current_sol, best_rand_sol)
        
        return best_rand_sol - np.multiply(A_vec, D)
    
    
    def whale_attack(self, current_sol, best_sol):
        
        D = np.linalg.norm(best_sol - current_sol)
        L = np.random.uniform(-1.0, 1.0, size=self.gene_length)
        
        return np.multiply(np.multiply(D,np.exp(self.whale_b*L)), np.cos(2.0*math.pi*L)) + best_sol
    
    def whale_constraints(self, gene_sol):
        
        phase_range = int(self.RIS_NumPhase / 2)
        # print("Befor cons : \n", gene_sol)
        for idx_c in range(self.gene_length):
            if idx_c < self.NumRISEle:
                if gene_sol[idx_c] < -phase_range:
                    gene_sol[idx_c]  = -phase_range
                elif gene_sol[idx_c] > phase_range:
                    gene_sol[idx_c]  = phase_range
            else:
                if gene_sol[idx_c] < 0.01:
                    gene_sol[idx_c]  = 5
                elif gene_sol[idx_c] > 40:
                    gene_sol[idx_c]  = 40
                    
        return gene_sol
    
    
    def whale_optimize(self, fitness_list, population):
        
        # ranked_sol = self._rank_solutions()
        sort_fitness = np.argsort(fitness_list) 
        best_sol = population[sort_fitness[int(2*self.population_group - 1)]]
        # print("best fit idx: ", sort_fitness[int(2*self.population_group - 1)], "\nbest fit : ", fitness_list[sort_fitness[int(2*self.population_group - 1)]])
        #include best solution in next generation solutions

        # print("a : ", self.whale_a)
        # for s in ranked_sol[1:]:
        for idx_wha in range(int(2*self.population_group - 1)):
            if np.random.uniform(0.0, 1.0) < 0.5:  
                # print("!!!Greater!!! ", idx_wha)                                    
                A = self.A_compute()
                norm_A = np.linalg.norm(A)
                # print("Norm : ", norm_A)                                 
                if norm_A < 1.0:                                                          
                    population[sort_fitness[idx_wha]] = self.Next_iter_position(population[sort_fitness[idx_wha]], best_sol, A)    
                    population[sort_fitness[idx_wha]] = np.round(population[sort_fitness[idx_wha]])                            
                else:                                                                     
                    ###select random sol                                                  
                    random_sol = population[np.random.randint(int(2*self.population_group))]       
                    population[sort_fitness[idx_wha]] = self.Next_iter_position(population[sort_fitness[idx_wha]], random_sol, A)    
                    # population[sort_fitness[idx_wha]] = np.round(population[sort_fitness[idx_wha]])                            
            else:
                # print("!!!Lower!!! ",  idx_wha)                                                                         
                population[sort_fitness[idx_wha]] = self.whale_attack(population[sort_fitness[idx_wha]], best_sol)
                # population[sort_fitness[idx_wha]] = np.round(population[sort_fitness[idx_wha]])
                
            # population[sort_fitness[idx_wha]] = np.round(population[sort_fitness[idx_wha]])
            population[sort_fitness[idx_wha]] = self.whale_constraints(population[sort_fitness[idx_wha]])
            fitness_list[sort_fitness[idx_wha]] = self.whale_fitness(population[sort_fitness[idx_wha]])                                
            # new_sols.append(self._constrain_solution(new_s))

        # self._sols = np.stack(new_s)
        self.whale_a -= self.whale_a_step
        
        return population, fitness_list
    
    
    # def whale_algorithm(self, init_population, init_fitness, maxiteration):
    def whale_algorithm(self, maxiteration):
        
        best_fitness = 0
        best_solution = np.zeros(self.gene_length)
        init_population = self.whale_generation_pop()
        init_fitness = np.zeros(int(2*self.population_group))
        
        # woa_each_iter_best = np.zeros(maxiteration)
        
        for idx_fit in range(int(2*self.population_group)):
            init_fitness[idx_fit] = self.whale_fitness(init_population[idx_fit])
        # print("WOA pop shape : ", init_population.shape, "\nWOA population: \n", init_population, "\nWOA fitness : \n", init_fitness)
        start_time = timeit.default_timer()
        for idx_iter_woa in range(maxiteration):
            
            # init_population, init_fitness = self.whale_optimize(init_fitness, init_population)
            init_population, init_fitness = self.whale_optimize(init_fitness, init_population)
            # print(init_fitness)
            
            # temp_sol = np.round(init_population[np.argmax(init_fitness)])
            # woa_each_iter_best[idx_iter_woa] = self.whale_fitness(temp_sol)
            
            # print("WOA population: \n", init_population, "\nWOA fitness : \n", init_fitness)
        
        end_start = timeit.default_timer()
        WOA_time = end_start - start_time
        # print(np.max(init_fitness))
        
        best_solution = np.round(init_population[np.argmax(init_fitness)])
        best_fitness =self.whale_fitness(best_solution)
        
        return best_fitness, best_solution, WOA_time#, woa_each_iter_best
        # return init_population, init_fitness, best_fitness, best_solution
    
    
    
    
    
    
    
    
    
    
    