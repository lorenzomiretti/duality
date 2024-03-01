# -*- coding: utf-8 -*-
"""
Code for "UL-DL Duality for Cell-free Massive MIMO with Per-AP Power and Information Constraints"
Output: Figure 2
Author: Lorenzo Miretti

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log2, log10, diag, eye
from numpy.linalg import norm, inv, pinv
from scipy.linalg import sqrtm

# My classes
from main_problem import Problem

def main():
    # Fixed problem parameters 
    L = 16                  # number of APs (must be a square number)
    N = 4                   # number of AP antennas 
    K = 16                  # number of UEs
    size  = 1000            # lenght of squared service area [m]
    P_max = 1000            # per-AP power constraints [mW]
    Q = 4                   # user-centric clusters size

    # Simulation parameters
    N_setups = 100
    R_list = np.linspace(2,4,8)
    prob_feas =  {"local": np.zeros(len(R_list)),"centr": np.zeros(len(R_list))}
    prob_feas_sum =  {"local": np.zeros(len(R_list)),"centr": np.zeros(len(R_list))}
    attempts_max = 4

    for r in range(len(R_list)):
        R = R_list[r]
        SINR_target = 2**R-1

        # Local CSI
        np.random.seed(0)
        CSI = 'local'   
        N_solved = 0 
        for n in range(N_setups):
            print("R = ", R, ", CSI = ", CSI, ", setup number = ", n+1, "/", N_setups)
            problem = Problem(L,K,N,size)
            problem.set_constraints(SINR_target*np.ones(K),P_max*np.ones(L),Q,CSI)
            solved = False
            kappa = 10
            attempts = 1
            while solved == False and attempts <= attempts_max:
                problem.solve(toll_fp = 1e-3,toll_psg = 1e-3*P_max,kappa = kappa,feas_only=True)
                print("Solved = ", problem.solved, ", feasible = ", problem.feasible)
                solved = problem.solved
                if solved == False:
                    kappa = 3*kappa
                    attempts += 1
            if problem.solved == True:
                N_solved += 1
                if problem.feasible == True:
                    prob_feas[CSI][r] += 1
                if problem.feasible_sum == True:
                    prob_feas_sum[CSI][r] += 1
        prob_feas[CSI][r] = prob_feas[CSI][r]/N_solved
        prob_feas_sum[CSI][r] = prob_feas_sum[CSI][r]/N_solved

        # Centralized CSI
        np.random.seed(0)
        CSI = 'centr'    
        N_solved = 0
        for n in range(N_setups):
            print("R = ", R, ", CSI = ", CSI, ", setup number = ", n+1, "/", N_setups)
            problem = Problem(L,K,N,size)
            problem.set_constraints(SINR_target*np.ones(K),P_max*np.ones(L),Q,CSI)
            solved = False
            kappa = 10
            attempts = 1
            while solved == False and attempts <= attempts_max:
                problem.solve(toll_fp = 1e-3,toll_psg = 1e-3*P_max,kappa = kappa,feas_only=True)
                print("Solved = ", problem.solved, ", feasible = ", problem.feasible)
                solved = problem.solved
                if solved == False:
                    kappa = 3*kappa
                    attempts += 1
            if problem.solved == True:
                N_solved += 1
                if problem.feasible == True:
                    prob_feas[CSI][r] += 1
                if problem.feasible_sum == True:
                    prob_feas_sum[CSI][r] += 1
        prob_feas[CSI][r] = prob_feas[CSI][r]/N_solved
        prob_feas_sum[CSI][r] = prob_feas_sum[CSI][r]/N_solved

    # Results
    plt.figure(figsize=(7, 4))
    fontSize = 15
    lwidth = 2
    msize = 10
    plt.plot(R_list,prob_feas['local'],'-bo',label='Local precoding',lw=lwidth,ms = msize)
    plt.plot(R_list,prob_feas_sum['local'],'--bo',label='Local precoding (sum power)',lw=lwidth,ms = msize)
    plt.plot(R_list,prob_feas['centr'],'-rd',label='Centralized precoding',lw=lwidth,ms = msize)
    plt.plot(R_list,prob_feas_sum['centr'],'--rd',label='Centralized precoding (sum power)',lw=lwidth,ms = msize)
    plt.xlabel('Minimum rate [b/s/Hz]',fontsize = fontSize)
    plt.ylabel('Probability of feasibility',fontsize = fontSize)
    plt.xticks(fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.grid()
    plt.legend(fontsize= fontSize)
    plt.show()

main()
