# -*- coding: utf-8 -*-
"""
Code for "UL-DL Duality for Cell-free Massive MIMO with Per-AP Power and Information Constraints"
Content: main routines used to generate the plots
Author: Lorenzo Miretti

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log10, diag, eye
from numpy.linalg import norm, inv
from scipy.linalg import sqrtm

class Problem:
    def __init__(self,L,K,N,size,N_sim = 100):
        self.L = int(sqrt(L))**2                    # number of APs (forced to be a square number)
        if self.L != L:
            print("Warning: L = %d is not a square number. Forced to L = %d." %(L, self.L))
        self.K = K                                  # number of UEs
        self.N = N                                  # number of AP antennas
        self.size = size                            # lenght of squared service area [m]               
        self.draw_network_setup()                   # Initialize network setup 
        self.draw_channel_realizations(N_sim)       # Initialize channel realizations for empirical evaluation of expectations

    def solve(self, toll_fp = 10**-3 ,toll_psg = 10**-4, iter_max_psg = 100, kappa = 3, feas_only = False):
        ''' toll_fp = tolerance for fixed point iterations
            toll_pg = tolerance projected gradient
            toll_feas = fesibility tolerance
            N_iter_max = maximum number of algorithm iterations
            kappa = diminishing step size constant (sqrt rule)
            earlystop = True if interested only in feasibility 
        '''
        self.solved = False
        self.feasible = False
        self.feasible_sum = True
        self.dual_obj = []
        self.powers = []
        # Initial dual uplink noise vector (such that first outer iteration focuses on sum power constraint)
        sig = np.ones(self.L)  
        # Initial power vector (such that first outer iteration has monotonic increase in sum power)                           
        p = self.SINR_target/(self.N*np.sum(self.Gamma, axis=1))                            
        # Lagrangian multipliers optimization (projected subgradient ascent)
        jj = 0
        while jj < iter_max_psg and self.solved == False:
            # Dual uplink power control (fixed point iteration)
            ii = 0
            self.solved_inner = False
            while self.solved_inner == False:     
                # Beamforming optimization
                parameters = self.optimize_beamforming_parameters(p,sig)
                interf, signal, noise = self.compute_channel_coefficients(p,sig,parameters)
                SINR = self.compute_SINR(interf,signal,noise,p,sig)  
                # Power control
                p_old = p
                p = np.array([self.SINR_target[k]*p[k]/SINR[k] for k in range(self.K)])
                SINR = self.compute_SINR(interf,signal,noise,p,sig)
                
                # Early stop if unfeasibility under sum power constraint (first outer iteration) is detected
                if jj == 0 and np.sum(p) > np.sum(self.P_max):  
                    self.solved_inner = True  
                    self.feasible_sum = False
                elif abs(sum(p)-sum(p_old))/sum(p_old) < toll_fp:
                    self.solved_inner = True  
                ii += 1
    
            # Dual objective 
            self.dual_obj.append(np.sum(p)-np.sum((sig-np.ones(self.L))*self.P_max))
            # Downlink powers
            self.powers.append(self.compute_downlink_powers(interf,signal,noise,p,sig,SINR))

            # Monitor progress
            dual_gap = sum(self.powers[jj])-self.dual_obj[jj]
            print('Dual: %.4f' % self.dual_obj[jj], '| surrogate dual gap: %.4f' % abs(dual_gap), '| max power gap: %.4f' % max(self.powers[jj] - self.P_max), '| max SINR gap: %.4f' % max(self.SINR_target-SINR))

            # Projected subgradient step
            sub_grad = self.powers[jj] - self.P_max
            sig = sig + kappa/sqrt(jj+1)*sub_grad/norm(sub_grad)
            sig = np.maximum(sig,np.ones(self.L))

            # Stopping condition
            if np.max(sub_grad) < toll_psg and (feas_only == True or dual_gap < toll_psg):
                self.feasible = True
                self.solved = True 

            # Early stop if unfeasibility is detected: 
            # - dual exceeds the sum of power constraints (upper bound on optimal sum power)
            # - SINR constraints unfeasible under arbitrary power (spectral radius check) from failed convergence of fixed point iteration
            if self.dual_obj[jj] > np.sum(self.P_max):
                self.feasible = False
                self.solved = True
            jj += 1
        self.p_opt = p
        self.sig_opt = sig
        
        
    def compute_downlink_powers(self,interf,signal,noise,p,sig,SINR):
        noise_UE = noise.T @ sig
        D = diag(signal/np.array(SINR))
        B = interf
        q = inv(D-B) @ inv(diag(noise_UE)) @ (D-B.T) @ p
        powers = noise @ q 
        return powers

    def compute_SINR(self,interf,signal,noise,p,sig):
        noise_UE = noise.T @ sig
        SINR = [p[k]*signal[k]/((interf[:,k]) @ p + noise_UE[k]) for k in range(self.K)]
        return SINR
        
    def compute_channel_coefficients(self,p,sig,parameters):
        ''' Numerically evaluate the equivalent channel coefficients for the UatF bound.
        ''' 
        interf = np.zeros((self.K,self.K))
        signal = np.zeros(self.K,dtype=complex)
        noise = np.zeros((self.L,self.K)) 
        for n in range(self.N_sim):
            # Compute combiners
            H_hat = self.H_hat_list[n] 
            V = self.beamformers(H_hat,p,sig,parameters)
            # Equivalent channel
            H_eq = np.zeros((self.K,self.K),dtype=complex)
            # Compute channel coefficients
            for l in range(self.L):
                H_eq += H_hat[l] @ V[l] 
                noise[l,:] += np.square(norm(V[l],axis=0))/self.N_sim   
            interf += np.square(np.abs(H_eq))/self.N_sim
            signal += diag(H_eq)/self.N_sim
        signal = np.square(np.abs(signal))  
        noise_CSI = self.Psi @ noise
        interf = interf-np.diag(signal) + noise_CSI
        return interf, signal, noise 

    def beamformers(self,H_hat,p,sig,parameters):
        L = self.L
        K = self.K
        N = self.N
        Q = self.Q

        if self.CSI == 'centr':    
            """ Centralized MMSE combining"""
            V_list = [np.zeros((N,K),dtype=complex) for _ in range(L)]
            for k in range(K):
                H_k = np.zeros((K,Q*N),dtype=complex)
                Psi_k = np.zeros((Q*N,Q*N),dtype=complex)
                Sig_k = np.zeros(Q*N,dtype=complex)
                for q in range(Q):
                    l = self.UE_clusters[k][q]
                    H_k[:,q*N:(q+1)*N] = H_hat[l]
                    Psi_q = p @ self.Psi[:,l]
                    Psi_k[q*N:(q+1)*N,q*N:(q+1)*N] = Psi_q*eye(N)
                    Sig_k[q*N:(q+1)*N] = sig[l]
                V_MMSE = inv(herm(H_k)@ diag(p) @ H_k + Psi_k + diag(Sig_k)) @ herm(H_k) @ diag(sqrt(p))
                for q in range(Q):
                    l = self.UE_clusters[k][q]
                    V_list[l][:,k] = V_MMSE[q*N:(q+1)*N,k]   
                   
        elif self.CSI == 'local':
            """ Local TMMSE combining """
            C_list = parameters
            V_list = [] 
            for l in range(L):
                H = H_hat[l]
                Psi_l = p @ self.Psi[:,l]
                V_LMMSE = inv(herm(H)@ diag(p) @ H + Psi_l*eye(N) + sig[l]*eye(N)) @ herm(H) @ diag(sqrt(p)) 
                V_list.append(V_LMMSE @ C_list[l]) 
        
        return V_list

    def optimize_beamforming_parameters(self,p,sig):
        if self.CSI == 'centr':    
            return None
        elif self.CSI == 'local':
            """ Computation of optimal parameters for LTMMSE combining """
            # Shorthands
            K = self.K
            L = self.L
            N = self.N
            Q = self.Q
            N_sim = self.N_sim
            # Compute regularized and normalized covariance matrices
            Pi_list = [] 
            for l in range(L):
                Pi_l = np.zeros((K,K),dtype=complex)
                Psi_l = p @ self.Psi[:,l]
                for n in range(N_sim):
                    H_hat = diag(sqrt(p))@ self.H_hat_list[n][l] 
                    Pi_l += H_hat @ inv(herm(H_hat) @ H_hat + Psi_l*eye(N) + sig[l]*eye(N)) @ herm(H_hat) 
                Pi_list.append(Pi_l / N_sim)
            # Compute large scale fading decoding coefficients 
            # Initialize some useful variables
            C_list = [np.zeros((K,K),dtype=complex) for _ in range(L)]
            B = np.zeros((K*Q,K), dtype = complex)
            for l in range(Q):
                B[K*l:K*(l+1),:] = np.eye(K,dtype=complex)
            # Solve system for every UE k
            for k in range(K):
                A = np.zeros((K*Q,K*Q),dtype=complex)
                for l in range(Q):
                    for j in range(Q):
                        if j == l:
                            A[K*l:K*(l+1),K*l:K*(l+1)] = np.eye(K,dtype=complex)
                        else:
                            A[K*l:K*(l+1),K*j:K*(j+1)] = Pi_list[self.UE_clusters[k][j]]
                c_k = inv(A) @ B[:,k]
                # Write coefficient onto C_list
                for q in range(Q):
                    l = self.UE_clusters[k][q] 
                    C_list[l][:,k] = c_k[K*q:K*(q+1)]
            return C_list
        
    def set_constraints(self,SINR_target,P_max,Q,CSI):
        self.SINR_target = SINR_target
        self.P_max = P_max
        self.Q = Q
        self.CSI = CSI
    
        # Compute user-centric cooperation clusters. Each user is served by the Q strongest APs.
        Gamma_sorted_indexes = np.argsort(self.Gamma,axis=1)
        UE_clusters = Gamma_sorted_indexes[:,-Q:]
        AP_clusters = [[] for _ in range(self.L)]
        for k in range(self.K):
            for q in range(Q):
                AP_clusters[UE_clusters[k,q]].append(k)
        self.UE_clusters = UE_clusters
        self.AP_clusters = AP_clusters
        
        # Compute channel estimates 
        H_hat_list = []
        for n in range(self.N_sim):
            H = self.H_list[n]
            # initialize channel estimates with channel means
            H_hat = [np.zeros((self.K,self.N),dtype=complex) for _ in range(self.L)]
            # store instantaneous realizations if UE k is served by AP l
            for l in range(self.L): 
                for k in AP_clusters[l]: 
                    H_hat[l][k,:]= H[l][k,:] 
            H_hat_list.append(H_hat)
        self.H_hat_list = H_hat_list

        # Compute error covariances (scaled identities, so we keep only the scaling)
        self.Psi = np.copy(self.Gamma)
        for l in range(self.L):
            for k in AP_clusters[l]:
                self.Psi[k,l] = 0

    def draw_network_setup(self):
        """ Square grid of APs, uniformly drawn UEs. 
            Path-loss model: 3GPP TR 38.901, NLoS UMi - street canyon
        """
        # Carrier frequency (GHz)
        f_c = 3.7
        # Pathloss exponent
        PL_exp = 3.53
        # Average channel gain in dB at a reference distance of 1 meter 
        Gamma_const = -22.4 
        # Minimum distance (i.e., the difference in height) between AP and UE in meters
        d_min = 10
        # Shadow fading std deviation [dB]
        sig_SF = 7.82
        # Bandwidth [Hz]
        B = 100*10**6
        # Noise figure [dB]
        noiseFigure = 7
        # Noise power [dBm]
        N0 = -174 + 10*log10(B) + noiseFigure

        # Generate AP positions (meters, Cartesian coordinates) 
        L = self.L
        K = self.K
        size = self.size
        x = np.linspace(0,size,int(sqrt(L))+1)   # "cell" boundaries on the x axis
        y = np.linspace(0,size,int(sqrt(L))+1)   # "cell" boundaries on the y axis 
        x_AP = (x[:-1]+x[1:])/2
        y_AP = (y[:-1]+y[1:])/2
        pos_APs = []
        for i in range(int(sqrt(L))):
            for j in range(int(sqrt(L))):
                pos = np.array([x_AP[i],y_AP[j]])
                pos_APs.append(pos)

        # Generate UE positions (meters, Cartesian coordinates)
        pos_UEs = [size*np.random.rand(2) for _ in range(K)]
        
        # UEs-APs distances (including d_min height difference), used for path loss computation
        dist = np.zeros((K,L))
        for k in range(K):
            for l in range(L):
                    dist[k,l]= np.sqrt(norm(pos_UEs[k]-pos_APs[l])**2 + d_min**2)

        # Uncorrelated shadow fading
        SF = sig_SF*np.random.randn(K,L)

        # UEs-UEs distances, used for computating shadow fading covariance matrices
        delta = np.zeros((K,K))
        for k in range(K):
            for j in range(K):
                delta[k,j] = norm(pos_UEs[k]-pos_UEs[j])

        # Correlated shadow fading (for each AP)
        for l in range(L):
            CovSF = np.power(2,-delta/13)  # SF covariance matrix (normalized by SF variance)
            SF[:,l] = sqrtm(CovSF) @ SF[:,l]

        # Channel gain (normalized by noise power)
        GammadB = Gamma_const - PL_exp*10*log10(dist) - 21.3*log10(f_c) - SF - N0

        # Store positions and path loss
        self.pos_UEs = pos_UEs
        self.pos_APs = pos_APs
        self.Gamma = 10**(GammadB/10)

    def draw_channel_realizations(self,N_sim):
        # Draw and store list of N_sim channel realizations
        self.H_list = []
        for _ in range(N_sim):
            # List of channel matrices, one for each AP. Rayleigh fading model.
            H = []
            for l in range(self.L):
                H_iid = 1/sqrt(2)*(np.random.standard_normal((self.K,self.N))+1j*np.random.standard_normal((self.K,self.N))) 
                H.append(diag(sqrt(self.Gamma[:,l])) @ H_iid)
            self.H_list.append(H)
        self.N_sim = N_sim

    def plot_network_setup(self):
        plt.figure()
        for k in range(self.K):
            if k == 0:
                plt.plot(self.pos_UEs[k][0],self.pos_UEs[k][1],'or',label="UE")
            else:
                plt.plot(self.pos_UEs[k][0],self.pos_UEs[k][1],'or')
        for l in range(self.L):
            if l == 0:
                plt.plot(self.pos_APs[l][0],self.pos_APs[l][1],'vb',label="AP")
            else:
                plt.plot(self.pos_APs[l][0],self.pos_APs[l][1],'vb')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('square')
        plt.xlim(0,self.size)
        plt.ylim(0,self.size)
        x = np.linspace(0,self.size,int(sqrt(self.L))+1)   # "cell" boundaries on the x axis
        y = np.linspace(0,self.size,int(sqrt(self.L))+1)   # "cell" boundaries on the y axis
        plt.xticks(x)
        plt.yticks(y)
        plt.legend()
        plt.grid()
        plt.show()

def herm(x):
    return x.conj().T






