# -*- coding: utf-8 -*-
"""
Code for "UL-DL Duality for Cell-free Massive MIMO with Per-AP Power and Information Constraints"
Output: Figure 3
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
    SINR_target = 2**3.5-1  # minimum SINR requirement
    CSI = 'local'           # information constraint

    # Example 1
    np.random.seed(4)
    CSI = 'local'   
    problem = Problem(L,K,N,size)
    problem.set_constraints(SINR_target*np.ones(K),P_max*np.ones(L),Q,CSI)
    kappa = 10
    iter_max = 20
    problem.solve(toll_fp = 1e-5,toll_psg = 1e-1,kappa = kappa, iter_max_psg = iter_max, feas_only=False)
    print("Solved = ", problem.solved, ", feasible = ", problem.feasible)
    dual_obj_1 = np.array(problem.dual_obj)
    max_power_1 = np.array([np.max(powers) for powers in problem.powers])

    # Example 2
    np.random.seed(4)
    CSI = 'local'   
    problem = Problem(L,K,N,size)
    problem.set_constraints(SINR_target*np.ones(K),P_max*np.ones(L),Q,CSI)
    kappa = 17
    iter_max = 20
    problem.solve(toll_fp = 1e-5,toll_psg = 1e-1,kappa = kappa, iter_max_psg = iter_max, feas_only=False)
    print("Solved = ", problem.solved, ", feasible = ", problem.feasible)
    dual_obj_2 = np.array(problem.dual_obj)
    max_power_2 = np.array([np.max(powers) for powers in problem.powers])

    # Plots
    # Dual objective
    plt.figure(figsize=(7, 4))
    fontSize = 15
    lwidth = 2
    msize = 10
    plt.plot(range(1,iter_max+1),10*log10(dual_obj_1),'b',label="$\\alpha = 10$",lw=lwidth)
    plt.plot(range(1,iter_max+1),10*log10(dual_obj_2),'--r',label="$\\alpha = 17$",lw=lwidth)
    plt.xlabel('Iteration number',fontsize = fontSize)
    plt.ylabel('Dual objective [dBm]',fontsize = fontSize)
    plt.xticks(fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.grid()
    plt.legend(fontsize= fontSize)

    # Maximum power
    plt.figure(figsize=(7, 4))
    fontSize = 15
    lwidth = 2
    msize = 10
    plt.plot(range(1,iter_max+1),10*log10(max_power_1),'b',label="$\\alpha = 10$",lw=lwidth)
    plt.plot(range(1,iter_max+1),10*log10(max_power_2),'--r',label="$\\alpha = 17$",lw=lwidth)
    plt.xlabel('Iteration number',fontsize = fontSize)
    plt.ylabel('Maximum power [dBm]',fontsize = fontSize)
    plt.xticks(range(1,iter_max+1),fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.grid()
    plt.legend(fontsize= fontSize)
    plt.show()

    

main()
