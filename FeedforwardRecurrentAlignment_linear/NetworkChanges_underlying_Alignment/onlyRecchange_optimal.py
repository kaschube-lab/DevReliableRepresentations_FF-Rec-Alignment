#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:55:46 2023

@author: sigridtragenap
"""
import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.decomposition import PCA
import collections

#append parent directory to python path
#easier local imports
#from networks_definition import LinearRecurrentNetwork, create_W
import Analysis.Alignment_funs as AlgnFunc
import tools.helper_funs as hf

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["figure.figsize"]=[2.5, 2.5]

import simulate_update_funcs as suf



#%% run simulations
n_neuron=200
n_stim=16
n_trial=15
n_events=n_trial*n_stim
run_stab=False

n_sim_total=200

simulation_params={
    'run_stab': run_stab,
    'n_neuron': n_neuron,
    'R': 0.85,
    'sigma_spont': 1.5,
    'tuning_strength': 1,
    'sigma_noise_evoked': 1.5,
    'sigma_time_evoked': 0.1,
    'n_trial' :n_trial ,
    'n_stim' : n_stim,
    'n_dim_spont': 20,
    'n_samples' :1e3 ,
    'ev_input_mode': 'Gauss_dim',  ##
    'noise_mode_ev': 'isotropic', ##
    'n_dim_evoked' : 10,    #
    'dim_sp_input': 20,        ###
    #'n_events_stab_intv' : 10,
    }

seeds_naive = np.arange(0,n_sim_total)
seeds_inputs = seeds_naive + np.random.randint(500,1000, n_sim_total)


key_file = "results_simulation/model_RecOptChange{}_Optimal.npz"



n_sim_total=5
for i in range(n_sim_total):

    res_Align_all=[]
    res_Align_first=[]


    #INitialize
    network_naive = suf.construct_network_naive(
        simulation_params,
        seeds_naive[i])

    #create static FF input
    evoked_input_l2_naive = suf.construct_evoked_input(
            simulation_params,
            input_seed=seeds_inputs[i],
            base_patterns_input=None)

    #run naive network
    data_naive = suf.run_SingleNetwork(simulation_params,
                                   evoked_input_l2=evoked_input_l2_naive,
                                   network_naive=network_naive,
                                   cov_only=True)

    data_naive_initial = data_naive.copy()

    evoked_input_l2_new = evoked_input_l2_naive.copy()
    data_new = data_naive.copy()


    res_Align_all.append(AlgnFunc.get_Alignment_all(network_naive, evoked_input_l2_new))
    res_Align_first.append(AlgnFunc.get_Alignment_first(network_naive, evoked_input_l2_new))

    res=[]
    for iupdate in tqdm(range(1)):
        #update recurrent
        network_updated = suf.update_recurrent(
                              network_naive,
                              data_new,
                              rule='optimal',
                              alpha=1.25)  #1.25

        #update FF
        evoked_input_l2_new = suf.update_cov_inputs(
                              network_naive,
                              data_new,
                              simulation_params,
                              rule='euler_recurrentEV',
                              alpha=0)


        #run experienced network
        data_new = suf.run_SingleNetwork(simulation_params,
                                        evoked_input_l2=evoked_input_l2_new,
                                        network_naive=network_updated,
                                        cov_only=True)

        #check alignment
        res_Align_all.append(AlgnFunc.get_Alignment_all(network_updated, evoked_input_l2_new))
        res_Align_first.append(AlgnFunc.get_Alignment_first(network_updated, evoked_input_l2_new))


        network_naive = network_updated

    plt.plot(res)
    plt.xlabel("Update steps")
    plt.ylabel("FF-Rec Alignment")
    plt.title("Joint learning of FF and Rec")



    np.savez(key_file.format(
        seeds_naive[i]),
        data_naive=data_naive_initial,
        data_exp=data_new,
        align_all=np.asarray(res_Align_all),
        align_first=np.asarray(res_Align_first),
        )

    spE = data_new['spont_patterns']
    evE = data_new['evoked_patterns'].reshape(
        -1,n_neuron)

    spN = data_naive_initial['spont_patterns']
    evN = data_naive_initial['evoked_patterns'].reshape(
        -1,n_neuron)
    _, _, ev_var_e = AlgnFunc.plot_overlap_A_in_B(evE,spE)
    plt.show()

