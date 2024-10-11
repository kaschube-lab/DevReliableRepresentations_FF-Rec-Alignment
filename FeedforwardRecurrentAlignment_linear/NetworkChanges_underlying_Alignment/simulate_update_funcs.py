#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:43:26 2023

@author: sigridtragenap
"""

import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.decomposition import PCA


#append parent directory to python path
#easier local imports
from tools.networks_definition import LinearRecurrentNetwork, create_W
import Analysis.Alignment_funs as AlgnFunc
import tools.helper_funs as hf

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["figure.figsize"]=[2.5, 2.5]

def tuning_curve_sines(NSTIM, noise_discount=1,stim_max=360, mode='sinus_ho'):
    stims=np.deg2rad(np.linspace(0,stim_max,NSTIM))
    if mode=='sinus_ho':
            responses = 0
            for isin in range(2,NSTIM*2,1):
                strength_ho = np.exp(-(isin-2)*noise_discount)
                responses += strength_ho*np.sin(isin*(stims+np.random.rand()*np.pi*2))
    return responses

def construct_evoked_input(
        simulation_params,
        input_seed,
        base_patterns_input=None,  #neurons, i, take these or random
        ):

    #print("TEST INSIDE")
    #extract params
    n_neuron=simulation_params['n_neuron']
    ev_input_mode = simulation_params['ev_input_mode']
    tuning_strength=simulation_params['tuning_strength']
    n_stim = simulation_params['n_stim']
    n_trial = simulation_params['n_trial']
    n_events = n_trial*n_stim
    if ev_input_mode=='Gauss_dim':
        dim_inputs = simulation_params['n_dim_evoked']
        #only create stimulus ensembles
        C_input = hf.create_n_dim_gauss(dim_inputs,
                                        base_patterns=base_patterns_input,
                                        n_neurons=n_neuron)

        tuned_comp = hf.sample_from_normal(C_input,
                                            n_samples=n_events)

        evoked_input_direct=tuned_comp*tuning_strength


        #print(np.min(evoked_input_direct), tuning_strength)

        all_inputs={
            'evoked_input_direct': evoked_input_direct,
            'cov_noise':np.eye(n_neuron),
            'raw_inputstrials': evoked_input_direct,
            'cov_signal': C_input,
                }
        return all_inputs


    evoked_trial_noise=simulation_params['sigma_noise_evoked']

    ev_noise_mode = simulation_params['noise_mode_ev']


    if base_patterns_input is None:
        base_patterns_input = hf.construct_onb(n_neuron, seed=input_seed)

    if ev_input_mode=='grating':
        #arrange stimuli from 0 to 360
        alphas=np.arange(n_stim)/n_stim*np.pi*2
        stim_input = np.sin(2*alphas)[:,None]*base_patterns_input[:,0][None,:] + np.cos(
                            2*alphas)[:,None]*base_patterns_input[:,1][None,:]
        tuned_comp=np.repeat(stim_input, n_trial, axis=0)
        tuned_comp /= linalg.norm(tuned_comp, axis=1)[:,None]
    if ev_input_mode=='grating_distorted':
        distortion = simulation_params['tuning_distortion']
        onb_random = hf.construct_onb(n_neuron, seed=input_seed*10+10)
        #arrange stimuli from 0 to 360
        alphas=np.arange(n_stim)/n_stim*np.pi*2
        stim_input = np.sin(2*alphas)[:,None]*base_patterns_input[:,0][None,:] + np.cos(
                            2*alphas)[:,None]*base_patterns_input[:,1][None,:]
        stim_input = stim_input + distortion*onb_random[:,:n_stim].T
        tuned_comp=np.repeat(stim_input, n_trial, axis=0)
        #is this necesary: only to compare across network sizes
        tuned_comp /= linalg.norm(tuned_comp, axis=1)[:,None]
    if ev_input_mode=='grating_sinus_ho':
        strength_decay = simulation_params['ho_decay']
        alphas=np.arange(n_stim)/n_stim*np.pi*2
        # #is this necesary: only to compare across network sizes
        stim_input = np.zeros((n_stim, n_neuron))
        for ineur in range(n_neuron):
            stim_input[:,ineur]=tuning_curve_sines(n_stim,
                                                   noise_discount=strength_decay,
                                                   stim_max=360, mode='sinus_ho')

        tuned_comp=np.repeat(stim_input, n_trial, axis=0)
        #is this necesary: only to compare across network sizes
        tuned_comp -= tuned_comp.mean()
        tuned_comp /= np.var(tuned_comp, axis=0)[None,:]
        #tuned_comp /= np.var(tuned_comp, axis=1)[:,None]
        #transform into base components

        #print(tuned_comp.shape,base_patterns_input.shape)
        #need to reshape to its own components first??
        pca = PCA()
        tuned_comp=pca.fit_transform(tuned_comp)
        tuned_comp = np.dot(tuned_comp,base_patterns_input.T)#/100
        #print(np.linalg.norm(tuned_comp, axis=1).mean(), np.linalg.norm(tuned_comp, axis=1).shape)
        #print(tuned_comp.shape)
    if ev_input_mode=="directDim":
        dim_inputs = simulation_params['n_dim_evoked']
        #only create stimulus ensembles
        C_input = hf.create_n_dim_gauss(dim_inputs,
                                        base_patterns=base_patterns_input,
                                        n_neurons=n_neuron)


        stim_input = hf.sample_from_normal(C_input,
                                            n_samples=n_stim)
        tuned_comp=np.repeat(stim_input, n_trial, axis=0)
        #is this necesary: only to compare across network sizes
        tuned_comp /= linalg.norm(tuned_comp, axis=1)[:,None]


    #print(tuned_comp.shape, tuning.shape, n_stim, n_trial)

    #additional trial noise
    if ev_noise_mode=='isotropic':
        # additional isotropic gauss
        cov_noise = np.eye(n_neuron)
        noise_component=np.random.randn(n_events,n_neuron)

    if ev_noise_mode=='low_dim_random':
        dim_ev_noise = simulation_params['ev_noise_dim']
        #draw samples from any low-dim Gauss
        cov_noise = hf.create_n_dim_gauss(dim_ev_noise,
                                          n_neurons=n_neuron)
        noise_component = hf.sample_from_normal(cov_noise,
                                                seed=np.random.randint(500),
                                                n_samples=n_events)
    if ev_noise_mode=='low_dim_baseinput':
        dim_ev_noise = simulation_params['ev_noise_dim']
        sigma_tuned = np.cov(tuned_comp.reshape(-1, n_neuron).T)
        #use eig instead of eigh because sigma_tuned is underdetermined
        #attention: return eigenvalues and vectors in a different order
        #no inverse sorting needed if eig
        eigvals, eigvecs_sigma = linalg.eig(sigma_tuned)
        eigvals = eigvals.real
        order = np.argsort(eigvals)[::-1]
        eigvecs_sigma=eigvecs_sigma.real

        cov_noise = hf.create_n_dim_gauss(
            dim_ev_noise,
            n_neurons=n_neuron,
            base_patterns=eigvecs_sigma[:,order])
        noise_component = hf.sample_from_normal(cov_noise,
                                                seed=np.random.randint(500),
                                                n_samples=n_events)


    evoked_input_direct=tuned_comp*tuning_strength+(
                        evoked_trial_noise*noise_component)
    all_inputs={
        'evoked_input_direct': evoked_input_direct,
        'cov_noise':cov_noise,
        'raw_inputstrials': tuned_comp*tuning_strength,
            }
    return all_inputs

def construct_network_naive(simulation_params, naive_seed):
    n_neuron=simulation_params['n_neuron']
    R=simulation_params['R']
    #sample connectivities
    try:
        mean_weights = simulation_params['w_mean']
    except:
        mean_weights=0
    w_rec_naive=create_W(n_neuron, mean_connect=mean_weights,symmetric=True, var_conn=1,
    				    spectral_bound=R, seed=naive_seed)

    #get eigenvectors of exp. connectivity - sets ups aligned input drives
    eigvals_naive, eigvecs_naive = linalg.eigh(w_rec_naive)

    network_naive = LinearRecurrentNetwork(w_rec_naive)
    return network_naive

def run_SingleNetwork(simulation_params,
                      evoked_input_l2,
                      network_naive,
                      cov_only=False):

    run_stab=simulation_params['run_stab']
    n_neuron=simulation_params['n_neuron']
    noise_sigma_input_time=simulation_params['sigma_time_evoked']  #Amplitude time-dependant noise  sigma_noise
    dim_sp_noise=simulation_params['dim_sp_input']  # Input dimension of spont
    n_stim=simulation_params['n_stim']   #Number of stimuli
    n_trial=simulation_params['n_trial']
    n_events=n_trial*n_stim  #total number of events



    eigvals_naive, eigvecs_naive = linalg.eigh(network_naive.W_connect)
    cov_noise = hf.create_n_dim_gauss(
        dim_sp_noise,
        n_neurons=n_neuron,
        base_patterns=eigvecs_naive[:,::-1])
    spont_input_naive = hf.sample_from_normal(cov_noise,
                                            seed=np.random.randint(500),
                                            n_samples=n_events)

    # run early experiments
    data_naive={}

    #steady states
    #print(np.nanmin(evoked_input_l2['evoked_input_direct']))

    trial_inputs = evoked_input_l2['evoked_input_direct'].T
    res_evoked = network_naive.res_Input_mat(trial_inputs).T
    res_spont = network_naive.res_Input_mat(spont_input_naive.T).T
    data_naive["spont_patterns"]=res_spont
    data_naive["spont_input"]=spont_input_naive
    data_naive["evoked_patterns"]=res_evoked.reshape(n_trial, n_stim, -1,
                                                     order='F')

    if cov_only:
        data_naive["spont_cov_in"] = cov_noise.copy()
        data_naive["spont_cov_out"] = network_naive.transform_sigma_input(
            data_naive["spont_cov_in"])

        #evoked
        data_naive["evoked_cov_in"] = evoked_input_l2['cov_signal'].copy()
        data_naive["evoked_cov_out"] = network_naive.transform_sigma_input(
            data_naive["evoked_cov_in"])


    if run_stab:
        trial_nf_inputs = evoked_input_l2['raw_inputstrials']
        n_events_stab = trial_nf_inputs.shape[0]
        #time dependent solution
        noise_type_time=simulation_params['noise_type_time']

        #print(noise_type_time)
        if noise_type_time=='white':
            cov_noise_ev=None
        if noise_type_time=='input':
            cov_noise_ev=evoked_input_l2['cov_noise']

        dt=1
        t1=1
        t2=21
        idx1=int(t1/dt)
        idx2=int(t2/dt)
        delta_dist = int((t2-t1)/dt)

        res_evoked_t1 = np.zeros_like(trial_nf_inputs)
        res_evoked_t2 = np.zeros_like(trial_nf_inputs)
        res_Stability = np.empty(n_events_stab,)*np.nan
        for itrial in range(n_events_stab)[::10]:
             all_resp = network_naive.time_dep_input_dW_color(
                 trial_nf_inputs[itrial],
                 dt=dt, T=50,
                 sigma=noise_sigma_input_time,
                 cov_noise=cov_noise_ev)

             res_Stability[itrial]=np.corrcoef(all_resp[idx1],
                                   all_resp[idx2])[1,0]
             res_evoked_t1[itrial] = all_resp[idx1]
             res_evoked_t2[itrial] = all_resp[idx2]

             #glattere Variante
             C=np.corrcoef(all_resp)
             res_Stability[itrial]=np.mean(np.diag(C, k=delta_dist))

        data_naive["evoked_patterns_t1"]=res_evoked_t1.reshape(n_trial, n_stim, -1,
                                                         order='F')
        data_naive["evoked_patterns_t2"]=res_evoked_t2.reshape(n_trial, n_stim, -1,
                                                         order='F')
        data_naive["Intra-trial stability"]=res_Stability.reshape(n_trial, n_stim, -1,
                                                         order='F')
    #add input
    data_naive["evoked_input_l2"] = evoked_input_l2
    return data_naive


def update_recurrent(network_naive,
                     data_activity,
                     rule='euler',
                     alpha=0.1):


    eigvals_naive, eigvecs_naive = linalg.eigh(network_naive.W_connect)

    if rule=='euler_input':
        evoked_input = data_activity['evoked_input_l2']['evoked_input_direct']  #evoked_input_direct
        sigma_evoked_naive = np.cov(evoked_input.T)
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_naive)

        eigvecs_interpol = eigvecs_naive + alpha*eigvecs_evoked_n
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)

    if rule=='euler_covinput':
        sigma_evoked_input = data_activity['evoked_cov_in']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)

        eigvecs_interpol = eigvecs_naive + alpha*eigvecs_evoked_n
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)

    if rule=='euler2_covinput':
        sigma_evoked_input = data_activity['evoked_cov_in']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)

        eigvecs_interpol = (1-alpha)*eigvecs_naive + alpha*eigvecs_evoked_n
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)

    if rule=='euler_covoutput':
        sigma_evoked_output = data_activity['evoked_cov_out']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_output)

        eigvecs_interpol = eigvecs_naive + alpha*eigvecs_evoked_n
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)


    if rule=='euler2_covoutput':
        sigma_evoked_output = data_activity['evoked_cov_out']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_output)

        eigvecs_interpol = (1-alpha)*eigvecs_naive + alpha*eigvecs_evoked_n
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)

    if rule=='rotate_ouput':
        sigma_evoked_output = data_activity['evoked_cov_out']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_output)

        inv_eigvecs_rec = np.linalg.inv(eigvecs_naive)
        rot_matrix = eigvecs_evoked_n@inv_eigvecs_rec

        eigvecs_interpol = alpha*rot_matrix@eigvecs_naive
        eigvecs_interpol_onb,_ = np.linalg.qr(eigvecs_interpol)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_interpol_onb)

    if rule=='output_direct':
        sigma_evoked_output = data_activity['evoked_cov_out']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_output)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_evoked_n)

    if rule=='optimal':
        sigma_evoked_input = data_activity['evoked_cov_in']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_evoked_n)

    if rule=='optimal_output':
        sigma_evoked_input = data_activity['evoked_cov_out']
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)

        #build based on sigma_input eigenvectors
        w_rec_update = hf.construct_mat(eigvals_naive, eigvecs_evoked_n)

    network_updated = LinearRecurrentNetwork(w_rec_update)
    return network_updated

def update_inputs(network_naive,
                     data_activity,
                     sim_params,
                      rule='euler_recurrentEV',
                      alpha=0.1,
                      seed=None):

    eigvals_naive, eigvecs_naive = linalg.eigh(network_naive.W_connect)
    #eigvecs_naive[:,np.argsort(eigvals_naive)] = eigvecs_naive

    if rule=='euler_recurrentEV':
        evoked_input = data_activity['evoked_input_l2']['raw_inputstrials']  #evoked_input_direct
        sigma_evoked_naive = np.cov(evoked_input.T)
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_naive)
        #sort by eigenvalues
        #eigvecs_evoked_n[:,np.argsort(eigvals_evoked_n)] = eigvecs_evoked_n

        eigvecs_interpol = eigvecs_evoked_n + alpha*eigvecs_naive
        eigvecs_input_onb,_ = np.linalg.qr(eigvecs_interpol)
        #print(np.linalg.norm(eigvecs_input_onb, axis=0))

        #print("test")
        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_input_onb[:,::-1])

        #print(np.min(evoked_inputs_new['evoked_input_direct']))
        return evoked_inputs_new


    if rule=='euler_outputEV':
        evoked_input = data_activity['evoked_input_l2']['evoked_input_direct']
        sigma_evoked_naive = np.cov(evoked_input.T)
        _, eigvecs_evoked_n = linalg.eigh(sigma_evoked_naive)


        evoked_act= data_activity['evoked_patterns']
        evoked_act=evoked_act.reshape(-1,evoked_act.shape[-1])
        sigma_evoked_act = np.cov(evoked_act.T)
        _, eigvecs_act = linalg.eigh(sigma_evoked_act)

        eigvecs_interpol = eigvecs_evoked_n + alpha*eigvecs_act
        eigvecs_input_onb,_ = np.linalg.qr(eigvecs_interpol)

        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_input_onb[:,::-1])
        return evoked_inputs_new

    if rule=='direct':
        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_naive[:,::-1])
        return evoked_inputs_new


def update_cov_inputs(network_naive,
                     data_activity,
                     sim_params,
                      rule='euler_recurrentEV',
                      alpha=0.1,
                      seed=None):

    eigvals_naive, eigvecs_naive = linalg.eigh(network_naive.W_connect)
    #eigvecs_naive[:,np.argsort(eigvals_naive)] = eigvecs_naive

    if rule=='euler_recurrentEV':

        sigma_evoked_input = data_activity['evoked_cov_in']
        _, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)

        eigvecs_interpol = eigvecs_evoked_n + alpha*eigvecs_naive
        eigvecs_input_onb,_ = np.linalg.qr(eigvecs_interpol)
        #print(np.linalg.norm(eigvecs_input_onb, axis=0))

        #print("test")
        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_input_onb[:,::-1])

        #print(np.min(evoked_inputs_new['evoked_input_direct']))
        return evoked_inputs_new


    if rule=='euler_outputEV':
        sigma_evoked_input = data_activity['evoked_cov_in']
        _, eigvecs_evoked_n = linalg.eigh(sigma_evoked_input)


        sigma_evoked_act = data_activity['evoked_cov_out']
        _, eigvecs_act = linalg.eigh(sigma_evoked_act)

        eigvecs_interpol = eigvecs_evoked_n + alpha*eigvecs_act
        eigvecs_input_onb,_ = np.linalg.qr(eigvecs_interpol)

        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_input_onb[:,::-1])
        return evoked_inputs_new

    if rule=='optimal':
        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_naive[:,::-1])
        return evoked_inputs_new

    if rule=='optimal_output':
        sigma_evoked_act = data_activity['evoked_cov_out']
        _, eigvecs_act = linalg.eigh(sigma_evoked_act)

        evoked_inputs_new = construct_evoked_input(
                sim_params,
                input_seed=seed,
                base_patterns_input=eigvecs_act[:,::-1])
        return evoked_inputs_new

