#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:00:25 2022

@author: sigridtragenap
"""

#set path to general code directory
from os.path import abspath, sep, pardir
import sys
sys.path.append(abspath('') + sep + pardir  + sep + pardir )


import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.decomposition import PCA


#append parent directory to python path
#easier local imports
from networks_definition import LinearRecurrentNetwork, create_W
import Analysis.Alignment_funs as AlgnFunc
import tools.helper_funs as hf

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["figure.figsize"]=[2.5, 2.5]


#construct tuning curves
def responses_tc(phase, strength, amp_off=0,
                 NSTIM=180, mode='sinus', noise_strength=1,
                 noise_discount=1, stim_max=360):
    stims=np.deg2rad(np.linspace(0,stim_max,NSTIM))
    if mode=='sinus':
        responses = amp_off+strength*np.sin(2*(stims+phase))
    if mode=='sinus_add':
        responses = amp_off+strength*np.sin(2*(stims+phase),
                    ) + np.random.rand(NSTIM)*noise_strength

    if mode=='sinus_ho_old':
        #assert noise_discount<=1
        responses = 0
        for isin in range(2,NSTIM*2,2):
            strength_ho = np.exp(-(isin-2)*noise_discount)
            responses += strength_ho*np.sin(isin*(stims+np.random.rand()*np.pi*2))
        responses *= amp_off
    if mode=='sinus_ho':
            #assert noise_discount<=1
            responses = 0
            for isin in range(2,NSTIM*2,1):
                strength_ho = np.exp(-(isin-2)*noise_discount)
                responses += strength_ho*np.sin(isin*(stims+np.random.rand()*np.pi*2))
            responses *= amp_off
    if mode=='mises':
        responses = amp_off*(np.exp(strength*np.cos(2*(stims-phase)))/np.exp(strength))
        #/(2*np.pi*np.i0(strength))

    return responses

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
        all_inputs={
            'evoked_input_direct': evoked_input_direct,
            'cov_noise':np.eye(n_neuron),
            'raw_inputstrials': evoked_input_direct,
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

    #print(noise_component.shape, n_events, n_trial)
    evoked_input_direct=tuned_comp*tuning_strength+(
                        evoked_trial_noise*noise_component)
    all_inputs={
        'evoked_input_direct': evoked_input_direct,
        'cov_noise':cov_noise,
        'raw_inputstrials': tuned_comp*tuning_strength,
            }
    return all_inputs

def run_SingleNetwork(simulation_params,
                      evoked_input_l2,
                      network_naive):

    run_stab=simulation_params['run_stab']
    n_neuron=simulation_params['n_neuron']
    #R=simulation_params['R']
    #R_l4=simulation_params['R_l4']
    #sigma_spont=simulation_params['sigma_spont']  #Amplitude spont  sigma_s
    #tuning_strength=simulation_params['tuning_strength']  #Amplitude evoked  sigma_e
    #noise_sigma_input=simulation_params['sigma_noise_evoked']  #Amplitude time-imdependant noise  sigma_noise
    noise_sigma_input_time=simulation_params['sigma_time_evoked']  #Amplitude time-dependant noise  sigma_noise
    dim_sp_noise=simulation_params['dim_sp_input']  # Input dimension of spont
    #broadness_ev_noise=simulation_params['broadness_ev']
    n_stim=simulation_params['n_stim']   #Number of stimuli
    n_trial=simulation_params['n_trial']
    n_events=n_trial*n_stim  #total number of events
    #stab_intv = simulation_params['n_events_stab_intv']
    #n_events_stab = int(n_events/stab_intv)

    eigvals_naive, eigvecs_naive = linalg.eigh(network_naive.W_connect)
    #spont: set up for each L2 network individually
    #noise should be low dimensional noise fitting to network
    # create spont input cov from highdim gauss
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
    trial_inputs = evoked_input_l2['evoked_input_direct'].T
    res_evoked = network_naive.res_Input_mat(trial_inputs).T
    res_spont = network_naive.res_Input_mat(spont_input_naive.T).T
    data_naive["spont_patterns"]=res_spont
    data_naive["spont_input"]=spont_input_naive
    data_naive["evoked_patterns"]=res_evoked.reshape(n_trial, n_stim, -1,
                                                     order='F')
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


def run_Rec_change(simulation_params,
                        input_seed=0,
                        network_seed=1,
                        network_naive=None,
                        evoked_input_l2_naive=None,
                        construct_exp_mode='sigma_in'):


    #create naive network
    if network_naive is None:
        network_naive = construct_network_naive(
            simulation_params,
            network_seed)

    #create static FF input
    if evoked_input_l2_naive is None:
        evoked_input_l2_naive = construct_evoked_input(
            simulation_params,
            input_seed,
            base_patterns_input=None)

    #run naive network
    data_naive = run_SingleNetwork(simulation_params,
                                   evoked_input_l2=evoked_input_l2_naive,
                                   network_naive=network_naive)

    #create new conn
    if construct_exp_mode=="sigma_in":
        evoked_input_pats=evoked_input_l2_naive['evoked_input_direct']
        sigma_evoked_naive = np.cov(evoked_input_pats.T)
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_naive)

        #build based on sigma_input eigenvectors
        w_rec_exp = hf.construct_mat(network_naive.eigvals, eigvecs_evoked_n)

    network_experienced= LinearRecurrentNetwork(w_rec_exp)

    #run experienced network
    data_experienced = run_SingleNetwork(simulation_params,
                                   evoked_input_l2=evoked_input_l2_naive,
                                   network_naive=network_experienced)

    #pass
    return data_naive, data_experienced

def run_Dmodel_FFchange(simulation_params,
                        input_seed=0,
                        network_seed=1,
                        network_naive=None,
                        evoked_input_l2_naive=None,
                        construct_exp_inpmode='direct'):

    #create static network
    if network_naive is None:
        network_naive = construct_network_naive(
            simulation_params,
            network_seed)

    #create naive FF input
    if evoked_input_l2_naive is None:
        evoked_input_l2_naive = construct_evoked_input(
            simulation_params,
            input_seed,
            base_patterns_input=None)


    #run naive network
    data_naive = run_SingleNetwork(simulation_params,
                                   evoked_input_l2_naive,
                                   network_naive=network_naive)

    #create experienced FF input
    if construct_exp_inpmode =="direct":
        evoked_input_l2_exp= construct_evoked_input(
            simulation_params,
            input_seed,
            base_patterns_input=network_naive.eigvecs[:,::-1],
            )

    #run experienced network
    data_experienced = run_SingleNetwork(simulation_params,
                                   evoked_input_l2=evoked_input_l2_exp,
                                   network_naive=network_naive)


    #pass
    return data_naive, data_experienced
#construct network
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


if __name__ == '__main__':
    naive_seed=18
    n_neuron=200
    n_trial=20
    n_stim=20

    simulation_params={
        'run_stab': False,
        'n_neuron': n_neuron,
        'R': 0.85,
        'sigma_spont': 0.05,
        'tuning_strength': 1,
        'sigma_noise_evoked': 0.1,
        'sigma_time_evoked': 0.1,
        'broadness_ev': 0.25,
        'n_trial' :n_trial ,
        'n_stim' : n_stim,
        'n_dim_spont': 18,
        'n_samples' :1e3 ,
        'n_dim_evoked' : 5,
        'ev_input_mode': 'directDim',
        'noise_mode_ev': 'isotropic',
        'dim_sp_input': 15,
        #'n_events_stab_intv' : 10,
        }

    # dim_ev_noise = simulation_params['ev_noise_dim']
    # tuning_strength=simulation_params['tuning_strength']
    # evoked_trial_noise=simulation_params['sigma_noise_evoked']
    # ev_input_mode = simulation_params['ev_input_mode']
    # ev_noise_mode = simulation_params['noise_mode_ev']

    #Test Network
    network_naive = construct_network_naive(simulation_params, naive_seed)
    evoked_input_l2_naive = construct_evoked_input(
            simulation_params,
            input_seed=2,
            base_patterns_input=None,  #neurons, i, take these or random
            )

    data = run_SingleNetwork(simulation_params,
                          evoked_input_l2_naive,
                          network_naive)

    AlgnFunc.plot_overlap_A_in_B(data['evoked_patterns'].reshape(-1,n_neuron),
                            data['spont_patterns'])


    #Test input Reconstruction
    construct_exp_mode="sigma_in"

    evoked_input_pats=evoked_input_l2_naive['evoked_input_direct']
    #reAlign network
    if construct_exp_mode=="sigma_in":
        sigma_evoked_naive = np.cov(evoked_input_pats.T)
        eigvals_evoked_n, eigvecs_evoked_n = linalg.eigh(sigma_evoked_naive)

        #build based on sigma_input eigenvectors
        w_rec_exp = hf.construct_mat(network_naive.eigvals, eigvecs_evoked_n)

    network_experienced= LinearRecurrentNetwork(w_rec_exp)

    #run experienced network
    data_experienced = run_SingleNetwork(simulation_params,
                                   evoked_input_l2=evoked_input_l2_naive,
                                   network_naive=network_experienced)
    AlgnFunc.plot_overlap_A_in_B(data_experienced['evoked_patterns'].reshape(-1,n_neuron),
                            data_experienced['spont_patterns'])



    #full test of Rec change
    # dataN, dataE = run_Rec_change(simulation_params,
    #                         input_seed=0,
    #                         network_seed=18,
    #                         network_naive=None,
    #                         evoked_input_l2_naive=None,
    #                         construct_exp_mode='sigma_in')
    # AlgnFunc.plot_overlap_A_in_B(dataN['evoked_patterns'].reshape(-1,n_neuron),
    #                         dataN['spont_patterns'])
    # AlgnFunc.plot_overlap_A_in_B(dataE['evoked_patterns'].reshape(-1,n_neuron),
    #                         dataE['spont_patterns'])
    #plt.close()
    #trial-trial-corr-matrices
    # trial_data=dataN['evoked_patterns'].reshape(-1,n_neuron, order='F')
    # C=np.corrcoef(trial_data)
    # plt.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    # plt.show()


    #full test of FF change
    dataN, dataE = run_Dmodel_FFchange(simulation_params,
                            input_seed=0,
                            network_seed=1,
                            network_naive=None,
                            evoked_input_l2_naive=None)

    AlgnFunc.plot_overlap_A_in_B(dataN['evoked_patterns'].reshape(-1,n_neuron),
                            dataN['spont_patterns'])
    AlgnFunc.plot_overlap_A_in_B(dataE['evoked_patterns'].reshape(-1,n_neuron),
                            dataE['spont_patterns'])

