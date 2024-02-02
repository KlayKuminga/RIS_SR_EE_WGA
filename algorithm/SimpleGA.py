#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:54:10 2023

@author: dddd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:34:03 2023

@author: dddd
"""

import numpy as np
import math
import random
import timeit

import function_compute as FUNC

# seed = 1
# random.seed(seed)

class Simple_Genetic:
    
    def __init__(self, NumBSAnt, NumRISEle, NumUE, NumPUE, NumSUE, sigma2, population_size, RIS_bits, mutate_rate, L_spreadingfactor, P_Rmin, S_Rmin,
                 P_max, PTx_static, mu, SIC_dispation, RIS_static, PUE_static, SUE_static, B2R, P_B2U, P_R2U, S_B2U, S_R2U):
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
        
    def Phase_shift_decode(self, phase_gene):
        
        Phase_shift_real = np.exp(1j*(2*math.pi*phase_gene/(8)))
        
        return Phase_shift_real
    
    
    
    def Power_decode(self, power_gene):
        
        Power_real = np.zeros(self.NumPUE)
        for idx_pow in range(self.NumPUE):
            Power_real[idx_pow] = pow(10, (power_gene[idx_pow]/10))/pow(10,3)
        
        return Power_real
    
    
    
    def Gen_Population(self):
        
        gene_length = self.NumRISEle + self.NumPUE
        phase_range = int(self.RIS_NumPhase / 2)
        # print("PH :", phase_range, "\n Num phase : ", self.RIS_NumPhase)
        population = np.zeros([self.population_size, gene_length], dtype=np.int)
        
        for idx_pop in range(self.population_size):
            for idx_len in range(gene_length):
                
                if(idx_len >= self.NumRISEle):
                    population[idx_pop][idx_len] = random.randint(15, 30)
                
                else:
                    population[idx_pop][idx_len] = random.randint(-phase_range, phase_range)
                    
        return population
    
        
    def fitness(self, individual_gene):
        
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
                score += (-5)*(self.P_Rmin - Primary_throughput_UE[idx_fp])
                
        for idx_fs in range(self.NumSUE):
            if( Secondary_throughput_UE[idx_fs] < self.S_Rmin):
                score += (-10)*(self.S_Rmin - Secondary_throughput_UE[idx_fs])
        
        
        temp_obj = self.Funcall.obj_func(Primary_throughput, Secondary_throughput, Power_consumption)
        score += temp_obj
        
        return score#,temp_obj
    
    def fitness_original(self, individual_gene):
        
        score = 0
        
        
        phase_gene = individual_gene[0:self.NumRISEle]
        power_gene = individual_gene[self.NumRISEle:]
        
        phase_decode = self.Phase_shift_decode(phase_gene)
        power_decode = self.Power_decode(power_gene)
        
        # if(sum(power_decode) > pow(10, (self.P_max/10))/pow(10,3)):
        #     score += -5
        
        Primary_channel = self.Funcall.Total_channel_conbime(np.diag(phase_decode), self.B2R, self.P_R2U, self.P_B2U)
        Secondary_channel = self.Funcall.Total_channel_conbime(np.diag(phase_decode), self.B2R, self.S_R2U, self.S_B2U)
        Primary_channel_minus = self.Funcall.Total_channel_minus(np.diag(phase_decode), self.B2R, self.P_R2U, self.P_B2U)
        
        Primary_throughput, Primary_throughput_UE = self.Funcall.primary_throughput(power_decode, Primary_channel, Primary_channel_minus)
        Secondary_throughput, Secondary_throughput_UE = self.Funcall.secondary_throughput(power_decode, Secondary_channel, self.L_spreadingfactor)
        Power_consumption = self.Funcall.power_consumption(power_decode, self.PTx_static, self.mu, self.SIC_dispation, self.RIS_static, self.PUE_static, self.SUE_static)
        
        # for idx_fp in range(self.NumPUE):
        #     if( Primary_throughput_UE[idx_fp] < self.P_Rmin):
        #         score += (-5)*(self.P_Rmin - Primary_throughput_UE[idx_fp])
                
        # for idx_fs in range(self.NumSUE):
        #     if( Secondary_throughput_UE[idx_fs] < self.S_Rmin):
        #         score += (-10)*(self.S_Rmin - Secondary_throughput_UE[idx_fs])
        
        
        temp_obj = self.Funcall.obj_func(Primary_throughput, Secondary_throughput, Power_consumption)
        # score += temp_obj
        
        return temp_obj
    
    def simple_selection_Parents(self, fitness_list):
        
        location = 0
        rand = random.random()        
        current_fitness_sum = 0.0
        temp_fit_list = np.zeros(self.population_size)
        
        min_fit = np.min(fitness_list)
        temp_fit_list = fitness_list + min_fit
        fitness_sum = np.sum(temp_fit_list)
        
        for idx_sel in range(fitness_list.size):
            current_fitness_sum += (temp_fit_list[idx_sel] / fitness_sum)
            
            if( rand < current_fitness_sum):
                location = idx_sel
                break        
        # print("Selection Location : ", location)
        return location
    
    
    
    def simple_crossover(self, father_gene, mother_gene):
        
        location_cross = int(random.random() * len(father_gene))
        child_cross = np.append(father_gene[0:location_cross], mother_gene[location_cross:])
        
        # print("Cross location : ", location_cross)
        
        return child_cross
    
    # def crossover2(self, father, mother):
        
    #     best_fitness=float('-inf')
    #     child=np.zeros(self.NumRISEle + self.NumPUE)
    #     # print('father_len',len(father))
        
    #     for i in range(father.size):
    #         current_child=np.zeros(self.NumRISEle + self.NumPUE)
    #         current_child=np.append(father[0:i],mother[i:])
    #         # print('current_child:',current_child.shape)
    #         current_fitness = self.fitness(current_child)
    #         if current_fitness>best_fitness:
    #             best_fitness=current_fitness
    #             # change:add copy
    #             child=current_child.copy()
        
    #     return child
    
    # def crossover3_mutation(self, sort_fitness, fitness_list, population):
        
        
    #     # best_fitness=float('-inf')
    #     # childs_gene = np.zeors([2*self.population_group, gene_length], dtype=np.int)
    #     father = population[sort_fitness[self.population_size - 1]]
    #     # print('father_len',len(father))
    #     idx_count_cro = 0
        
    #     for idx_cro in range( int(self.population_group) ):
            
    #         mother = population[sort_fitness[(self.population_size - 2) - idx_cro]]
    #         index_divide = int(random.random()*(self.gene_length))
    #         # print("cross index : ", index_divide)
    #         population[sort_fitness[idx_count_cro]] = np.append(father[0:index_divide],mother[index_divide:])
    #         # population[sort_fitness[idx_cro]] = self.crossover2(father, mother)
            
    #         # print('current_child:',current_child.shape)
    #         population[sort_fitness[idx_count_cro]] = self.mutation(population[sort_fitness[idx_count_cro]])
    #         fitness_list[sort_fitness[idx_count_cro]] = self.fitness(population[sort_fitness[idx_count_cro]])
            
    #         population[sort_fitness[idx_count_cro + 1]] = np.append(mother[0:index_divide],father[index_divide:])
            
    #         population[sort_fitness[idx_count_cro + 1]] = self.mutation(population[sort_fitness[idx_count_cro + 1]])
    #         fitness_list[sort_fitness[idx_count_cro + 1]] = self.fitness(population[sort_fitness[idx_count_cro + 1]])
            
    #         idx_count_cro += 2
            
    #         # if current_fitness>best_fitness:
    #         #     best_fitness=current_fitness
    #         #     # change:add copy
    #         #     child=current_child.copy()
        
    #     return population, fitness_list
    
    
    def simple_mutation(self, child_gene_bin):
        
        child_mutate_bin = child_gene_bin
        
        if(random.random() < self.mutate_rate):
            index = int(random.random()*len(child_mutate_bin))
            if ( index >= self.NumRISEle):
                child_mutate_bin[index] = random.randint(15, 40)
            else:
                child_mutate_bin[index] = random.randint(-int(self.RIS_NumPhase / 2), int(self.RIS_NumPhase / 2))
                
        return child_mutate_bin
    
    
    # def random_population(self, population, sort_fitness, fitness_list, Numrandom):
        
        
    #     phase_range = int(self.RIS_NumPhase / 2)
    #     Numrandom2 = int(2*Numrandom)
        
    #     for idx_ran in range(Numrandom):
    #         for idx_var in range(self.gene_length):
                
    #             if(idx_var < self.NumRISEle):
    #                 population[sort_fitness[idx_ran + Numrandom2]][idx_var] = random.randint(-phase_range + 1, phase_range)
    #                 # fitness_list[sort_fitness[idx_ran + Numrandom2]] = self.fitness(population[sort_fitness[idx_ran + Numrandom2]])
                    
    #             else:
    #                 population[sort_fitness[idx_ran + Numrandom2]][idx_var] = random.randint(15,35)
    #                 # fitness_list[sort_fitness[idx_ran + Numrandom2]] = self.fitness(population[sort_fitness[idx_ran + Numrandom2]])
            
    #         fitness_list[sort_fitness[idx_ran + Numrandom2]] = self.fitness(population[sort_fitness[idx_ran + Numrandom2]])
        
    #     return population, fitness_list
    
    
    def Simple_Gene_Algorithm(self, Max_itreation):
        
        fitness_list = np.zeros(self.population_size)
        simple_best_fitness = float('-inf')
        Last_best_fitness = 0
        best_gene = np.zeros((self.NumRISEle + self.NumPUE))
        population = self.Gen_Population()
        Numrandom = self.population_group
        # print("Population first : \n", population, "\nfiness list first: \n", fitness_list)
        
        ga_each_iter_best = np.zeros(Max_itreation)
        
        for idx_fit in range(self.population_size):
            fitness_list[idx_fit] = self.fitness(population[idx_fit])
        
        start = timeit.default_timer()
        for idx_iter in range(Max_itreation):
            
            
            # print("Population before : \n", population, "\nfiness list before: \n", fitness_list)
            for id_gene in population:
                current_fitness = self.fitness(id_gene)
            # Sort_fitness = np.argsort(fitness_list)
            # temp_best_idx = Sort_fitness[self.population_size-1]
            
                if(current_fitness > simple_best_fitness):
                    simple_best_fitness = current_fitness
                    best_gene = id_gene.copy()
                
            
                
                for idx in range(self.population_size):
                    # print("Population inner before : \n", population, "\nfiness list inner before: \n", fitness_list)
                    fitness_list[idx] = self.fitness(population[idx])
                    father_gene = population[self.simple_selection_Parents(fitness_list)]
                    mother_gene = population[self.simple_selection_Parents(fitness_list)]
                    child_from_bestparents = self.simple_crossover(father_gene, mother_gene)
                    child_from_bestparents = self.simple_mutation(child_from_bestparents)
                    population[idx] = child_from_bestparents
                    # print("index: ", idx ,", Population inner after : \n", population, "\nfiness list inner after: \n", fitness_list)
                    
                # print("Population after : \n", population, "\nfiness list after: \n", fitness_list)
                
            # print(np.max(fitness_list))
            ga_each_iter_best[idx_iter] = self.fitness_original(best_gene)

        if(np.max(fitness_list) > simple_best_fitness):
            simple_best_fitness = np.max(fitness_list)
            best_gene = population[np.argmax(fitness_list)].copy()
        
        end = timeit.default_timer()
        simple_GA_time = end-start
        simple_best_fitness = self.fitness_original(best_gene)
        # print(simple_best_fitness)
        phase_best = self.Phase_shift_decode(best_gene[0:self.NumRISEle])
        power_best = self.Power_decode(best_gene[self.NumRISEle:])
                
        
        
        return simple_best_fitness, best_gene, phase_best, power_best, simple_GA_time, ga_each_iter_best