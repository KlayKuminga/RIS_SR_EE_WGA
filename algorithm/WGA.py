#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:47:43 2023

@author: dddd
"""

import numpy as np
import math
import random
import timeit

import function_compute as FUNC

class Discrete_whale_genetic:
    
    def __init__(self, NumBSAnt, NumRISEle, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, P_Rmin, S_Rmin,
                 P_max, PTx_static, mu, SIC_dispation, RIS_static, PUE_static, SUE_static, B2R, P_B2U, P_R2U, S_B2U, S_R2U, whale_a_max, whale_b):
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
        self.phase_range = int(self.RIS_NumPhase / 2)
        self.mutate_rate = mutate_rate
        self.population_group = int(1/5*population_size)
        self.gene_length = self.NumRISEle + self.NumPUE
        self.whale_a_max = whale_a_max
        self.whale_a = whale_a_max
        # self.whale_a_step = whale_a_step
        self.whale_b = whale_b
        # self.c1 = constant_1
        
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
        population = np.zeros([self.population_size, self.gene_length], dtype=np.float32)
        
        for idx_pop in range(self.population_size):
            for idx_len in range(self.gene_length):
                
                if(idx_len >= self.NumRISEle):
                    population[idx_pop][idx_len] = random.randint(15, 30)
                
                else:
                    population[idx_pop][idx_len] = random.randint(-phase_range, phase_range)
                    
        return population
    
    
    def whale_generation_pop_special(self):
        
        power_min = 15
        power_max = 35
        power_range = power_max - power_min
        pop_each_level = 10
        pop_power_level = int(self.population_size / pop_each_level)
        power_level_num = int((power_range)/pop_power_level)
        # pop_phase_level = int(self.population_size / 10)
        phase_level = int(self.RIS_NumPhase / 2)
        phase_level_num = int(self.RIS_NumPhase / phase_level)
        # print("PH :", phase_range, "\n Num phase : ", self.RIS_NumPhase)
        population = np.zeros([self.population_size, self.gene_length], dtype=np.float32)
        
        temp_pop_power = 0
        temp_pop_phase = 0
        
        for idx_level in range(pop_power_level):
            # print("phase level : ", temp_pop_phase)
            if temp_pop_phase > (phase_level - 1):
                temp_pop_phase = 0
            for idx_pop in range(pop_each_level):
                
                for idx_len in range(self.gene_length):
                    
                    if(idx_len <= self.NumRISEle):
                        population[idx_pop + idx_level*pop_each_level][idx_len] = random.randint(-phase_level + temp_pop_phase * phase_level_num, (-phase_level + 2) + temp_pop_phase * phase_level_num)
                    
                    else:
                        population[idx_pop + idx_level*pop_each_level][idx_len] = random.randint(power_min + idx_level*(power_level_num + 1), power_min + (power_level_num + 1) + idx_level*(power_level_num + 1))
                
            temp_pop_power += 1
            temp_pop_phase += 1
            
        return population
    
    def whale_generation_pop_latin(self):
        
        power_min = 15
        power_max = 35
        power_range = power_max - power_min
        pop_each_level = 10
        pop_power_level = 4
        power_level_num = int((power_range)/pop_power_level)
        # pop_phase_level = int(self.population_size / 10)
        phase_level = int(self.RIS_NumPhase / 2)
        phase_level_num = int(self.RIS_NumPhase / phase_level)
        # print("PH :", phase_range, "\n Num phase : ", self.RIS_NumPhase)
        population = np.zeros([self.population_size, self.gene_length], dtype=np.float32)
        
        idx_latin = 0
        # idx_latin1 = []
        idx_latin2 = []
        # idx_latin1 = [0, 1, 2, 3]
        idx_latin2 = [0, 1, 2, 3]

        for idx_pop in range(self.population_size):
            for idx_len in range(self.gene_length):
                
                if len(idx_latin2) > 1:
                    # print("L1", len(idx_latin2))
                    idx_latin = idx_latin2[random.randint(0, len(idx_latin2) - 1)]
                    
                elif len(idx_latin2) == 1:
                    # print("L2", len(idx_latin2))
                    idx_latin = idx_latin2[0]
                
                else:
                    # print("L3", len(idx_latin2))
                    idx_latin2 = [ idx_latin1 for idx_latin1 in range(phase_level)]
                    idx_latin = idx_latin2[random.randint(0, len(idx_latin2) - 1)]
                    
                # print(idx_latin)

                if(idx_len < self.NumRISEle):
                    population[idx_pop][idx_len] = random.randint(-phase_level + idx_latin * phase_level_num, (-phase_level + 1) + idx_latin * phase_level_num)
                
                else:
                    population[idx_pop][idx_len] = random.randint(power_min + idx_latin * power_level_num, power_min + (power_level_num - 1) + idx_latin * power_level_num)
                    
                idx_latin2.remove(idx_latin)
        
            
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
        
        return best_rand_sol - self.c1 * np.multiply(A_vec, D)
    
    
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
    
    def cross_idx_gen(self):
        
        random_idx1 = int(random.random()*(self.gene_length))
        random_idx2 = int(random.random()*(self.gene_length))
        random_arr = np.sort(np.array([random_idx1, random_idx2]))
        
        return random_arr[0], random_arr[1]
    
    def gen_random_gene(self):
        random_gene = np.zeros(self.gene_length)
        
        for idx_r in range(self.gene_length):
            
            if idx_r <self.NumRISEle:
                random_gene[idx_r] =  random.randint(-self.phase_range, self.phase_range-1)
            else:
                random_gene[idx_r] =  random.randint(15, 40)
        return random_gene
    
    def whale_crossover_random(self, fitness_list, population):
        
        temp_num = int(2 * self.population_group)
        # best_fitness=float('-inf')
        # childs_gene = np.zeors([2*self.population_group, gene_length], dtype=np.int)
        sort_fitness = np.argsort(fitness_list)
        father = population[sort_fitness[self.population_size - 1]]
        # print('father_len',len(father))
        temp_cross0 = 0
        temp_cross1 = 0
        idx_count_cro = 0
        
        for idx_cro in range( int(self.population_group) ):
            
            mother1 = population[np.random.randint(self.population_size - 1)]
            mother2 = self.gen_random_gene()
            index_divide1, index_divide2 = self.cross_idx_gen()
            # print("cross index : ", index_divide)
            
            # Child for one dot crossover with random choose gene from population
            population[sort_fitness[idx_count_cro]] = np.append(father[0:index_divide2],mother1[index_divide2:])
            
            # print('current_child:',current_child.shape)
            population[sort_fitness[idx_count_cro]] = self.mutation(population[sort_fitness[idx_count_cro]])
            fitness_list[sort_fitness[idx_count_cro]] = self.whale_fitness(population[sort_fitness[idx_count_cro]])
            
            # Child for two dot crossover with random generate gene
            temp_cross0 = np.append(father[0:index_divide1], mother2[index_divide1:index_divide2])
            population[sort_fitness[idx_count_cro + temp_num]] = np.append(temp_cross0, father[index_divide2:])
            
            population[sort_fitness[idx_count_cro + temp_num]] = self.mutation(population[sort_fitness[idx_count_cro + temp_num]])
            fitness_list[sort_fitness[idx_count_cro + temp_num]] = self.whale_fitness(population[sort_fitness[idx_count_cro + temp_num]])
            
            # Another child for one dot crossover with random choose gene from population
            population[sort_fitness[idx_count_cro + 1]] = np.append(mother1[0:index_divide2],father[index_divide2:])
            
            population[sort_fitness[idx_count_cro + 1]] = self.mutation(population[sort_fitness[idx_count_cro + 1]])
            fitness_list[sort_fitness[idx_count_cro + 1]] = self.whale_fitness(population[sort_fitness[idx_count_cro + 1]])
            
            # Another child for two dot crossover with random generate gene
            temp_cross1 = np.append(mother2[0:index_divide1], father[index_divide1:index_divide2])
            population[sort_fitness[idx_count_cro + temp_num + 1]] = np.append(temp_cross1, mother2[index_divide2:])
            
            population[sort_fitness[idx_count_cro + temp_num + 1]] = self.mutation(population[sort_fitness[idx_count_cro + temp_num + 1]])
            fitness_list[sort_fitness[idx_count_cro + temp_num + 1]] = self.whale_fitness(population[sort_fitness[idx_count_cro + temp_num + 1]])
            
            idx_count_cro += 2
            
            # if current_fitness>best_fitness:
            #     best_fitness=current_fitness
            #     # change:add copy
            #     child=current_child.copy()
        
        return population, fitness_list
    
    def whale_crossover_single(self, single_whale, random_whale):
        
        
        return_whale = np.zeros(self.gene_length)
        random_idx_single = int(random.random()*(self.gene_length))
        single_whale = np.round(single_whale)
        random_whale = np.round(random_whale)
        
        crossover_whale1 = np.append(single_whale[0:random_idx_single], random_whale[random_idx_single:])
        crossover_whale2 = np.append(random_whale[0:random_idx_single], single_whale[random_idx_single:])
        
        if self.whale_fitness(crossover_whale1) > self.whale_fitness(crossover_whale2):
            return_whale = crossover_whale1
        else:
            return_whale = crossover_whale2
        
            
        
        return return_whale
    
    
    def whale_crossover_single_random(self, single_whale):
        
        
        return_whale = np.zeros(self.gene_length)
        randint_whale = np.zeros(self.gene_length)
        randint_whale = self.gen_random_gene()
        
        random_idx_single = int(random.random()*(self.gene_length))
        single_whale = np.round(single_whale)
        
        crossover_whale1 = np.append(single_whale[0:random_idx_single], randint_whale[random_idx_single:])
        crossover_whale2 = np.append(randint_whale[0:random_idx_single], single_whale[random_idx_single:])
        
        if self.whale_fitness(crossover_whale1) > self.whale_fitness(crossover_whale2):
            return_whale = crossover_whale1
        else:
            return_whale = crossover_whale2
        
            
        
        return return_whale
    
    
    def whale_crossover3_mutation(self, fitness_list, population):
        
        
        # best_fitness=float('-inf')
        # childs_gene = np.zeors([2*self.population_group, gene_length], dtype=np.int)
        temp_num = int(2 * self.population_group)
        
        sort_fitness = np.argsort(fitness_list)
        father = population[sort_fitness[self.population_size - 1]]
        # print('father_len',len(father))
        temp_cross0 = 0
        temp_cross1 = 0
        idx_count_cro = 0
        
        for idx_cro in range( int(self.population_group) ):
            
            mother = population[sort_fitness[(self.population_size - 2) - idx_cro]]
            index_divide1, index_divide2 = self.cross_idx_gen()
            # print("cross index : ", index_divide)
            population[sort_fitness[idx_count_cro]] = np.append(father[0:index_divide2],mother[index_divide2:])
            # population[sort_fitness[idx_cro]] = self.crossover2(father, mother)
            
            # print('current_child:',current_child.shape)
            population[sort_fitness[idx_count_cro]] = self.mutation(population[sort_fitness[idx_count_cro]])
            fitness_list[sort_fitness[idx_count_cro]] = self.whale_fitness(population[sort_fitness[idx_count_cro]])
            
            
            temp_cross0 = np.append(father[0:index_divide1], mother[index_divide1:index_divide2])
            population[sort_fitness[idx_count_cro + temp_num]] = np.append(temp_cross0, father[index_divide2:])
            
            population[sort_fitness[idx_count_cro + temp_num]] = self.mutation(population[sort_fitness[idx_count_cro + temp_num]])
            fitness_list[sort_fitness[idx_count_cro + temp_num]] = self.whale_fitness(population[sort_fitness[idx_count_cro + temp_num]])
            
            
            population[sort_fitness[idx_count_cro + 1]] = np.append(mother[0:index_divide2],father[index_divide2:])
            
            population[sort_fitness[idx_count_cro + 1]] = self.mutation(population[sort_fitness[idx_count_cro + 1]])
            fitness_list[sort_fitness[idx_count_cro + 1]] = self.whale_fitness(population[sort_fitness[idx_count_cro + 1]])
            
            
            temp_cross1 = np.append(mother[0:index_divide1], father[index_divide1:index_divide2])
            population[sort_fitness[idx_count_cro + temp_num + 1]] = np.append(temp_cross1, mother[index_divide2:])
            
            population[sort_fitness[idx_count_cro + temp_num + 1]] = self.mutation(population[sort_fitness[idx_count_cro + temp_num + 1]])
            fitness_list[sort_fitness[idx_count_cro + temp_num + 1]] = self.whale_fitness(population[sort_fitness[idx_count_cro + temp_num + 1]])
            
            idx_count_cro += 2
            
            # if current_fitness>best_fitness:
            #     best_fitness=current_fitness
            #     # change:add copy
            #     child=current_child.copy()
        
        return population, fitness_list
    
    def mutation(self, child_gene_bin):
        
        child_mutate_bin = child_gene_bin
        for idx_mu in range(int((self.NumRISEle + self.NumPUE) / 4)):
            if(random.random() < self.mutate_rate):
                index = int(random.random()*len(child_mutate_bin))
                if ( index >= self.NumRISEle):
                    child_mutate_bin[index] = random.randint(15, 35)
                else:
                    child_mutate_bin[index] = random.randint(-int(self.RIS_NumPhase / 2), int(self.RIS_NumPhase / 2))
                
        return child_mutate_bin
    
    def whale_optimize_discrete(self, fitness_list, population, whale_iteration):
        
        half_iter = int((whale_iteration/2))
        
        for idx_woga in range(whale_iteration):
            # print("Iter whale best 1: ", np.max(fitness_list))
            sort_fitness = np.argsort(fitness_list) 
            best_sol = population[sort_fitness[self.population_size - 1]]
            # print("best fit idx: ", sort_fitness[int(2*self.population_group - 1)], "\nbest fit : ", fitness_list[sort_fitness[int(2*self.population_group - 1)]])
            #include best solution in next generation solutions
    
            # print("a : ", self.whale_a)
            # self.c1 = 1
            if idx_woga < half_iter:
                # self.c1 = 0.5 * (1 - np.cos(np.pi + (np.pi * idx_woga / whale_iteration)))**0.5
                self.c1 = 0.5 * (1 + np.cos(np.pi* idx_woga / whale_iteration))**0.5
            else :
                self.c1 = 0.5 * (1 - np.cos(np.pi + (np.pi * idx_woga / whale_iteration)))**0.5
            
            for idx_wha in range(self.population_size - 1):
                if np.random.uniform(0.0, 1.0) < 0.5:  
                    # print("!!!Greater!!! ", idx_wha)                                    
                    A = self.A_compute()
                    norm_A = np.linalg.norm(A)
                    # print("Norm ", idx_woga, " : ", norm_A)   
                    # print("===!!A!!===")                              
                    if norm_A < 1.0:
                        # print("===!!S!!===")                                               
                        population[sort_fitness[idx_wha]] = np.round(self.Next_iter_position(population[sort_fitness[idx_wha]], best_sol, A))
                        # population[sort_fitness[idx_wha]] = self.Next_iter_position(population[sort_fitness[idx_wha]], best_sol, A)                               
                    else:                                                                     
                        ###select random sol
                        # print("===!!R!!===")
                        # random_sol = population[np.random.randint(self.population_size - 1)]       
                        # population[sort_fitness[idx_wha]] = np.round(self.Next_iter_position(population[sort_fitness[idx_wha]], random_sol, A))
                        population[sort_fitness[idx_wha]] = self.whale_crossover_random_single(population[sort_fitness[idx_wha]], best_sol)
                        # population[sort_fitness[idx_wha]] = self.whale_crossover_random_single(population[sort_fitness[idx_wha]], random_sol)                                
                else:
                    # print("!!!--B--!!!")
                    # print("!!!Lower!!! ",  idx_wha)                                                                         
                    population[sort_fitness[idx_wha]] = np.round(self.whale_attack(population[sort_fitness[idx_wha]], best_sol))
                    # population[sort_fitness[idx_wha]] = self.whale_attack(population[sort_fitness[idx_wha]], best_sol)
                
                
                population[sort_fitness[idx_wha]] = self.whale_constraints(population[sort_fitness[idx_wha]])
                fitness_list[sort_fitness[idx_wha]] = self.whale_fitness(population[sort_fitness[idx_wha]])
                
    
            # self._sols = np.stack(new_s)
            self.whale_a = self.whale_a_max - self.whale_a_max * (idx_woga/whale_iteration)
            # print(self.whale_a)
            # print("Iter whale best 2: ", np.max(fitness_list))
            # continuous to discrete
            # print("Round fit bef: ", fitness_list[sort_fitness[self.population_size - 1]], "\n", population[sort_fitness[self.population_size - 1]])
            
            population[sort_fitness[self.population_size - 1]] = np.round(population[sort_fitness[self.population_size - 1]])
            fitness_list[sort_fitness[self.population_size - 1]] = self.whale_fitness(population[sort_fitness[self.population_size - 1]])
            # print("Iter num ", idx_woga, ": ",np.max(fitness_list))
            # print("Round fit aft: ", fitness_list[sort_fitness[self.population_size - 1]], "\n", population[sort_fitness[self.population_size - 1]])
            # print("Iter whale best 3: ", np.max(fitness_list))
        
        return population, fitness_list
    
    def whale_architec_optimize_discrete(self, fitness_list, population, iter_best, whale_iteration):
        
        half_iter = int((whale_iteration/2))
        
        for idx_woga in range(whale_iteration):
            # print("Iter whale best 1: ", np.max(fitness_list))
            sort_fitness = np.argsort(fitness_list) 
            best_sol = population[sort_fitness[self.population_size - 1]]
            # print("best fit idx: ", sort_fitness[int(2*self.population_group - 1)], "\nbest fit : ", fitness_list[sort_fitness[int(2*self.population_group - 1)]])
            #include best solution in next generation solutions
    
            # print("a : ", self.whale_a)
            # self.c1 = 1
            if idx_woga < half_iter:
                # self.c1 = 0.5 * (1 - np.cos(np.pi + (np.pi * idx_woga / whale_iteration)))**0.5
                self.c1 = 0.5 * (1 + np.cos(np.pi* idx_woga / whale_iteration))**0.5
            else :
                self.c1 = 0.5 * (1 - np.cos(np.pi + (np.pi * idx_woga / whale_iteration)))**0.5
            
            for idx_wha in range(self.population_size - 1):
                if np.random.uniform(0.0, 1.0) < 0.5:  
                    # print("!!!Greater!!! ", idx_wha)                                    
                    A = self.A_compute()
                    norm_A = np.linalg.norm(A)
                    # print("Norm ", idx_woga, " : ", norm_A)   
                    # print("===!!A!!===")                              
                    if norm_A < 1.0:
                        # print("===!!S!!===")                                               
                        population[sort_fitness[idx_wha]] = np.round(self.Next_iter_position(population[sort_fitness[idx_wha]], best_sol, A))
                        # population[sort_fitness[idx_wha]] = self.Next_iter_position(population[sort_fitness[idx_wha]], best_sol, A)                             
                    else:                                                                     
                        ###select random sol
                        # print("===!!R!!===")
                        # random_sol = population[np.random.randint(self.population_size - 1)]       
                        # population[sort_fitness[idx_wha]] = np.round(self.Next_iter_position(population[sort_fitness[idx_wha]], random_sol, A))
                        # population[sort_fitness[idx_wha]] = self.whale_crossover_random_single(population[sort_fitness[idx_wha]], best_sol)
                        # population[sort_fitness[idx_wha]] = self.whale_crossover_single(population[sort_fitness[idx_wha]], random_sol)
                        population[sort_fitness[idx_wha]] = self.whale_crossover_single_random(population[sort_fitness[idx_wha]])
                        population[sort_fitness[idx_wha]] = self.mutation(population[sort_fitness[idx_wha]])                                
                else:
                    # print("!!!--B--!!!")
                    # print("!!!Lower!!! ",  idx_wha)                                                                         
                    # population[sort_fitness[idx_wha]] = np.round(self.whale_attack(population[sort_fitness[idx_wha]], best_sol))
                    population[sort_fitness[idx_wha]] = self.whale_crossover_single(population[sort_fitness[idx_wha]], best_sol)
                    population[sort_fitness[idx_wha]] = self.mutation(population[sort_fitness[idx_wha]])
                
                
                population[sort_fitness[idx_wha]] = self.whale_constraints(population[sort_fitness[idx_wha]])
                fitness_list[sort_fitness[idx_wha]] = self.whale_fitness(population[sort_fitness[idx_wha]])
            
            iter_best[idx_woga] = np.max(fitness_list)
    
            # self._sols = np.stack(new_s)
            self.whale_a = self.whale_a_max - self.whale_a_max * (idx_woga/whale_iteration)
            # print(self.whale_a)

            # continuous to discrete
            # print("Round fit bef: ", fitness_list[sort_fitness[self.population_size - 1]], "\n", population[sort_fitness[self.population_size - 1]])
            
            population[sort_fitness[self.population_size - 1]] = np.round(population[sort_fitness[self.population_size - 1]])
            fitness_list[sort_fitness[self.population_size - 1]] = self.whale_fitness(population[sort_fitness[self.population_size - 1]])
            # print("Iter num ", idx_woga, ": ",np.max(fitness_list))

        
        return population, fitness_list, iter_best
    
    

    def discrete_whale_genetic_algorithm(self, maxiteration):
        
        best_fitness = 0
        best_solution = np.zeros(self.gene_length)
        # whale_population = self.whale_generation_pop()
        # whale_population = self.whale_generation_pop_special()
        whale_population = self.whale_generation_pop_latin()
        # print(whale_population)
        whale_fitness = np.zeros(self.population_size)
        DWOA_iter_best = np.zeros(maxiteration)
        
        for idx_fit in range(self.population_size):
            whale_fitness[idx_fit] = self.whale_fitness(whale_population[idx_fit])
        # print("WOA pop shape : ", init_population.shape, "\nWOA population: \n", init_population, "\nWOA fitness : \n", init_fitness)
        
        # for idx_iter_woa in range(maxiteration):
        start_time = timeit.default_timer()
        
        # whale_population, whale_fitness = self.whale_optimize_discrete(whale_fitness, whale_population, maxiteration)
        whale_population, whale_fitness, DWOA_iter_best = self.whale_architec_optimize_discrete(whale_fitness, whale_population, DWOA_iter_best, maxiteration)
        
            
            
        # print("WOA population: \n", init_population, "\nWOA fitness : \n", init_fitness)
        end_start = timeit.default_timer()
        DWOA_time = end_start - start_time
        best_fitness = np.max(whale_fitness)
        best_solution = np.round(whale_population[np.argmax(whale_fitness)])
        # best_fitness = self.whale_fitness(best_solution)
        # print(np.max(whale_fitness), " ", best_fitness)
        
        return best_fitness, best_solution, DWOA_time, DWOA_iter_best
        # return init_population, init_fitness, best_fitness, best_solution
    
    
    
    
    
    
    
    
    
    
    