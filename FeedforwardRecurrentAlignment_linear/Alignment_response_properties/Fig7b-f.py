#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:16:35 2022

@author: sigridtragenap
"""

#You need to set the python working directory to the main git folder
# in spyder, you can do this with the 'Project' functionality or as below

#Alternative method:
#from os.path import abspath, sep, pardir
#import sys
#sys.path.append(abspath('') + sep + pardir)


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


#import network functions
from tools.networks_definition import LinearRecurrentNetwork, create_W
import Analysis.Alignment_funs as AlgnFunc
import tools.helper_funs as hf

#imports nicer plotting settings/stylesheet
from tools.plotting_funcs import cm2inch, color_al, color_r

#%%  Network parameters and Setup

#general parameters
n_neuron=200
n_stim=16
n_trial=50
n_events=n_trial*n_stim
run_stab=True
noise_level_time=0.3
sigma_ttc=0.02
simulation_params={
    'run_stab': run_stab,
    'n_neuron': n_neuron,
    'R': 0.85,
    'sigma_spont': 1.5,
    'tuning_strength': 1,
    'sigma_noise_evoked': 1.5,
    'sigma_time_evoked': 0.1,
    'broadness_ev': 0.25,
    'n_trial' :n_trial ,
    'n_stim' : n_stim,
    'n_dim_spont': 20,
    'n_samples' :1e3 ,
    'n_dim_evoked' : 10,
    #'n_events_stab_intv' : 10,
    }

#%% Loop over networks to get statistics of influence of  FF-Rec Alignment on
#response properties such as trial-to-trial correlation, Intra-trial-stability
#Dimensionality, #Alignment

version='EV'

collect_alignmentJ=[]
collect_ttc=[]
collect_its=[]
collect_dim=[]
collect_alignment_spont=[]
for inetwork in range(5):
    #Set up network
    w_rec=create_W(n_neuron, mean_connect=0,
                         symmetric=True, var_conn=1,
    				    spectral_bound=simulation_params["R"],
                        seed=inetwork)
    eigvals_net, eigvecs_net = linalg.eigh(w_rec)

    reccurent_network = LinearRecurrentNetwork(w_rec)

    #Run statistics for Trial-trial corr, Intra-trial stability, ALignment
    if version=='EV':
        Stimensembles=eigvecs_net[:,-100:].T


    w_rec = reccurent_network.W_connect
    ndrive=len(Stimensembles)

    #get evoked responses
    drive_resp= reccurent_network.res_Input_mat(Stimensembles.T).T
    norms=linalg.norm(drive_resp, axis=1)
    drive_resp /= norms[:,None]

    #get spontanepus response
    C_in_spont= hf.create_n_dim_gauss(20,
                                   base_patterns=eigvecs_net[:,::-1],
                                   n_neurons=n_neuron)
    C_output = reccurent_network.transform_sigma_input(C_in_spont)
    spont_frames = hf.sample_from_normal(cov=C_output)


    #Spontaneous Alignment: compare evoked responses to spontaneous covariance
    res_drive_spont=np.diag(
        drive_resp@C_output@drive_resp.T)/np.trace(C_output)
    collect_alignment_spont.append(res_drive_spont)


    #Trial-to-Trial correlation
    res_ttc=np.empty(ndrive)*np.nan
    for ipat in range(ndrive):
        all_trials = hf.sample_from_normal(np.eye(n_neuron)*sigma_ttc,
                                            mean=Stimensembles[ipat],
                                            n_samples=100)

        norms=linalg.norm(all_trials, axis=1)
        all_trials /= norms[:,None]

        responses = reccurent_network.res_Input_mat(all_trials.T).T
        #patsi=responses-responses.mean(1)[:,None]
        C=np.corrcoef(responses)

        # plt.imshow(C,cmap='RdBu_r', vmin=-1, vmax=1)
        # plt.show()
        res_ttc[ipat]=np.nanmean(C[np.triu_indices(100, k=1)])

    collect_ttc.append(np.asarray(res_ttc))

    #intra-trial stability
    dt=0.1
    delta_dist=int(20/dt)
    res_its=np.empty(ndrive)*np.nan
    for ipat in range(ndrive):
        all_resp = reccurent_network.time_dep_input_dW(Stimensembles[ipat],
                                                            dt=dt, T=100,
                                                            sigma=noise_level_time)
        C=np.corrcoef(all_resp)

        res_its[ipat]=np.mean(np.diag(C, k=delta_dist))

    collect_its.append(np.asarray(res_its))

    #Alignment (anatomical)
    res_Rayleigh_align=np.diag(Stimensembles@w_rec@Stimensembles.T)
    collect_alignmentJ.append(np.asarray(res_Rayleigh_align))


    # Dimensionality
    eigbasis = reccurent_network.eigvecs
    for K in [10,]:
        dim_input = K
        x=np.arange(n_neuron)[::-1]
        res_dim=[]
        res_align_dim=[]
        for idx in np.arange(n_neuron//2):
            indices = np.roll(x, -1*idx)
            C_input = hf.create_n_dim_gauss(K, base_patterns=eigbasis[:,indices],
                                            n_neurons=n_neuron)

            C_output = reccurent_network.transform_sigma_input(C_input)
            eigvals, _ = linalg.eigh(C_output)
            res_dim.append(hf.dimensionality(eigvals))

            leading_comp=eigbasis[:,indices][:,0]
            res_align_dim.append(leading_comp@reccurent_network.W_connect@leading_comp.T)

    collect_dim.append(np.asarray(res_dim))


#%%
np.savez("results_theoretical_vN.npz",  ttc=collect_ttc,
         anatom=collect_alignmentJ, funct=collect_alignment_spont,
         its=collect_its ,
         dimensionality = collect_dim)


#%%

#reload results
data_theory=np.load("results_theoretical_vN.npz")
res_ttc=data_theory['ttc']
res_Rayleigh_align=data_theory['anatom']
res_drive_spont=data_theory['funct']
res_its = data_theory['its']
res_align_spont = data_theory['funct']
res_dim = data_theory['dimensionality']




#%% Examples for two stimuli
stimA = eigvecs_net[:,-1]
stimC = np.random.randn(n_neuron)
stimC /= linalg.norm(stimC)


T=120
dt=0.1
resT=[]
for i,drive in enumerate([stimA,  stimC]):
    print(i)
    rng = np.random.default_rng(seed=i*5+10)
    activity_time = reccurent_network.time_dep_input_dW(drive,
                                                        dt=dt, T=T,
                                                        sigma=noise_level_time)
    resT.append(activity_time)
time=np.arange(int(T/dt)+1)*dt
Nlines=1


n_drives=100
res_ttc_toy=[]
for i,drive in enumerate([stimA, stimC]):
    stim = np.copy(drive)
    #C_trials = stim.T@stim + np.eye(n_neuron)
    all_trials = hf.sample_from_normal(np.eye(n_neuron)*sigma_ttc, mean=stim,
                                       n_samples=n_drives)
    responses = reccurent_network.res_Input_mat(all_trials.T).T

    C=np.corrcoef(responses)
    av_trial_corr = np.nanmean(C[np.triu_indices(n_drives, k=1)])
    res_ttc_toy.append(C[np.triu_indices(n_drives, k=1)])

    Cinput=np.corrcoef(all_trials)
    #plt.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    av_trial_corrIn = Cinput[np.triu_indices(n_drives, k=1)]
    #res_in.append(av_trial_corrIn.mean())


#%%  Curves for Dimensionality and Alignment
#statistics over multiple networks

n_sims=25
res_sp_in=np.zeros((n_sims, n_neuron))
res_sp_out=np.zeros((n_sims, n_neuron))
res_evA_in=np.zeros((n_sims, n_neuron))
res_evA_out=np.zeros((n_sims, n_neuron))
res_evR_in=np.zeros((n_sims, n_neuron))
res_evR_out=np.zeros((n_sims, n_neuron))
res_A_overlap=np.zeros((n_sims, n_neuron))
res_R_overlap=np.zeros((n_sims, n_neuron))
for i in range(n_sims):
    #Set up network
    w_rec=create_W(n_neuron, mean_connect=0,
                         symmetric=True, var_conn=1,
    				    spectral_bound=simulation_params["R"],
                        seed=i*5)
    eigvals_net, eigvecs_net = linalg.eigh(w_rec)
    reccurent_network = LinearRecurrentNetwork(w_rec)
    #create spont input ensemble
    C_in_spont= hf.create_n_dim_gauss(simulation_params['n_dim_spont'],
                                   base_patterns=eigvecs_net[:,::-1],
                                   n_neurons=n_neuron)
    C_output = reccurent_network.transform_sigma_input(C_in_spont)
    samples_sp_in = hf.sample_from_normal(cov=C_in_spont, n_samples=int(1e4))
    samples_sp_out = hf.sample_from_normal(cov=C_output, n_samples=int(1e4))
    _,  sp_var_in = AlgnFunc.plot_PC_spectrum(samples_sp_in)
    plt.close()
    res_sp_in[i]=(sp_var_in)
    _,  sp_var_o = AlgnFunc.plot_PC_spectrum(samples_sp_out)
    plt.close()
    res_sp_out[i]=(sp_var_o)
    #Aligned evoked
    C_in_ev= hf.create_n_dim_gauss(simulation_params['n_dim_evoked'],
                                   base_patterns=eigvecs_net[:,::-1],
                                   n_neurons=n_neuron)
    C_output = reccurent_network.transform_sigma_input(C_in_ev)
    samples_ev_in = hf.sample_from_normal(cov=C_in_ev, n_samples=int(1e4))
    samples_ev_out = hf.sample_from_normal(cov=C_output, n_samples=int(1e4))
    _,  sp_var_in = AlgnFunc.plot_PC_spectrum(samples_ev_in)
    plt.close()
    res_evA_in[i]=(sp_var_in)
    _,  sp_var_o = AlgnFunc.plot_PC_spectrum(samples_ev_out)
    plt.close()
    res_evA_out[i]=(sp_var_o)
    _,_ ,  overlap= AlgnFunc.plot_overlap_A_in_B(samples_ev_out,
                                  samples_sp_out)
    plt.close()
    res_A_overlap[i]=(overlap)
    #random evoked
    C_in_ev= hf.create_n_dim_gauss(simulation_params['n_dim_evoked'],
                                   base_patterns=None,
                                   n_neurons=n_neuron)
    C_output = reccurent_network.transform_sigma_input(C_in_ev)
    samples_ev_in = hf.sample_from_normal(cov=C_in_ev, n_samples=int(1e4))
    samples_ev_out = hf.sample_from_normal(cov=C_output, n_samples=int(1e4))
    _,  sp_var_in = AlgnFunc.plot_PC_spectrum(samples_ev_in)
    plt.close()
    res_evR_in[i]=(sp_var_in)
    _,  sp_var_o = AlgnFunc.plot_PC_spectrum(samples_ev_out)
    plt.close()
    res_evR_out[i]=(sp_var_o)
    _,_ ,  overlap= AlgnFunc.plot_overlap_A_in_B(samples_ev_out,
                                  samples_sp_out)
    plt.close()
    res_R_overlap[i]=(overlap)



#%% Summary subplot

fig, axs = plt.subplot_mosaic("""ABCXD
                                 EFG.H""",
                constrained_layout=False,figsize=cm2inch(14.,4.5),
                sharey=False,sharex=False,
                gridspec_kw={'wspace':0.8, 'hspace':0.7,})
plt.subplots_adjust(bottom=0.15, right=0.98, left=0.08, top=0.9,
                 )

#Toy examples
axttc=axs['A']
axttc.hist(res_ttc_toy[0], bins=100, range=[-1,1], density=True, alpha=0.5,
           color=color_al,
           orientation="vertical")
axttc.hist(res_ttc_toy[1], bins=100, range=[-1,1], density=True,  alpha=0.5,
           color=color_r,
           orientation="vertical")
axttc.axvline(av_trial_corrIn.mean(), color='k', ls='--')
axttc.set_xlabel("Correlation\nacross trials", linespacing=0.8, fontsize=7)
axttc.set_ylabel("Frequency", fontsize=7)
axttc.tick_params(length=1.5, width=0.5, labelsize=7)
axttc.xaxis.set_label_coords(0.5, -0.25)
#axttc.set_xticks([0,5])
#axttc.set_xlim(xmin=None, xmax=5)
axttc.yaxis.set_label_coords(-0.3,0.5)

#its
axline=axs['B']
for i,color in enumerate([color_al, color_r, ]):
    sort_std = np.mean(np.abs(resT[i]), axis=0)
    sort_m = np.argsort(np.mean(resT[i], axis=0))[::-1]
    axline.plot(time, (resT[i]/sort_std)[:,sort_m][:,:Nlines], color=color,
            alpha=0.6, zorder=10-2*i)
axline.set_xlabel("Time", linespacing=0.8, fontsize=7)
axline.set_ylabel("Normalized  \nactivity  ", linespacing=0.8, fontsize=7)
axline.set_xticks([0,120])
axline.set_yticks([2,0,-2])
axline.yaxis.set_label_coords(-0.28, 0.5)
axline.xaxis.set_label_coords(0.5,-0.25)


axN=axs['C']

axN.plot(np.arange(n_neuron)+1,res_evA_out.mean(0), color=color_al,
         label='Aligned', lw=1)
axN.fill_between(np.arange(n_neuron)+1,
                 res_evA_out.mean(0) + res_evA_out.std(0),
                 res_evA_out.mean(0) - res_evA_out.std(0),
                 color=color_al, alpha=0.2)
axN.plot(np.arange(n_neuron)+1,res_evR_out.mean(0), color=color_r,
         label='Random',
          lw=1.)
axN.fill_between(np.arange(n_neuron)+1,
                 res_evR_out.mean(0) + res_evR_out.std(0),
                 res_evR_out.mean(0) - res_evR_out.std(0),
                 color=color_r, alpha=0.2)

axN.set_ylim(ymin=0, ymax=None);

axN.set_xticks([1,10,20], ['1', "", "20"])

axN.set_xlim(xmax=20, xmin=0.1)
axN.set_ylabel("Variance explained", linespacing=0.8)
axN.set_xlabel("PC Index")
#axN.yaxis.set_label_coords(-0.2,0.5)
axN.xaxis.set_label_coords(0.5,-0.25)
leg=axN.legend(frameon=False, handlelength=0.2, loc=(0.15,0.5), handletextpad=0.3,
            labelspacing=0, ncol=1, columnspacing=0.2,
            title='Inputs:    ')
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

#Spont input output transformation
axSI=axs['X']
color_ev=plt.cm.viridis(0.35)
color_sp=plt.cm.plasma(0.5)
x_pos = np.arange(n_neuron)+1
axSI.plot(x_pos,res_sp_out.mean(0), color=color_sp, lw=1,
         label='Activity', zorder=3)
axSI.fill_between(x_pos,
                 res_sp_out.mean(0) + res_sp_out.std(0),
                 res_sp_out.mean(0) - res_sp_out.std(0),
                 color=color_sp, alpha=0.4, edgecolor='none')

#Aligned
axSI.plot(x_pos,res_sp_in.mean(0), color='gray', lw=1,
         label='Broad inputs', zorder=2)
axSI.fill_between(x_pos,
                 res_sp_in.mean(0) + res_sp_in.std(0),
                 res_sp_in.mean(0) - res_sp_in.std(0),
                 color='gray', alpha=0.4, edgecolor='none')
#Fract. var. expl.
axSI.set_ylabel("Variance explained", linespacing=0.8)
axSI.set_ylim(ymin=0, ymax=None);

axSI.set_xticks([1,10,20], ['1', "", "20"])

axSI.set_xlim(xmax=20, xmin=0.1)
#axA.set_ylabel("Fract. variance  \nexpl.", linespacing=0.8)
axSI.set_xlabel("PC index")
axSI.yaxis.set_label_coords(-0.1,0.5)
axSI.xaxis.set_label_coords(0.5,-0.25)
leg=axSI.legend(frameon=False, handlelength=0.2, loc=(0.15,0.3), handletextpad=0.3,
            labelspacing=0, ncol=1, columnspacing=0.2,
            title='Spontaneous   \nmodel:   ')
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)



#Spont alignment
axA=axs['D']
color_ev=plt.cm.viridis(0.35)
color_sp=plt.cm.plasma(0.5)
x_pos = np.arange(n_neuron)+1
axA.plot(x_pos,res_sp_out.mean(0), color=color_sp, lw=1,
         label='Spont. act.', zorder=3)
axA.fill_between(x_pos,
                 res_sp_out.mean(0) + res_sp_out.std(0),
                 res_sp_out.mean(0) - res_sp_out.std(0),
                 color=color_sp, alpha=0.4, edgecolor='none')

#Aligned
axA.plot(x_pos,res_A_overlap.mean(0), color=color_al, lw=1,
         label='Alig. Input', zorder=2)
axA.fill_between(x_pos,
                 res_A_overlap.mean(0) + res_A_overlap.std(0),
                 res_A_overlap.mean(0) - res_A_overlap.std(0),
                 color=color_al, alpha=0.4, edgecolor='none')

#random
axA.plot(x_pos,res_R_overlap.mean(0), color=color_r, lw=1,
         label='Rnd. Input', zorder=1)
axA.fill_between(x_pos,
                 res_R_overlap.mean(0) + res_R_overlap.std(0),
                 res_R_overlap.mean(0) - res_R_overlap.std(0),
                 color=color_r, alpha=0.4, edgecolor='none')


axA.set_ylim(ymin=0, ymax=None);

axA.set_xticks([1,10,20], ['1', "", "20"])

axA.set_xlim(xmax=20, xmin=0.1)
axA.set_ylabel("Variance explained", linespacing=0.8)
axA.set_xlabel("Spont. PC")
axA.yaxis.set_label_coords(-0.1,0.5)
axA.xaxis.set_label_coords(0.5,-0.25)
leg=axA.legend(frameon=False, handlelength=0.2, loc=(0.15,0.4), handletextpad=0.3,
            labelspacing=0, ncol=1, columnspacing=0.2, fontsize=7)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)


#%%

s_marker=2
alpha_marker=0.1
linewidth_marker=0.3

## measure over ensemble
axttc=axs['E']
axttc.scatter(res_Rayleigh_align, res_ttc, color='k',
            s=s_marker, alpha=alpha_marker, linewidths=linewidth_marker)
#axttc.xlabel("Alignment \n FF recurrent  ", linespacing=0.8)
axttc.set_ylabel("Trial-to-trial\n correlation", linespacing=0.8)
axttc.set_xlim(xmin=0)
axttc.set_yticks([0,1])
axttc.set_xticks([0,1])

axits=axs['F']
axits.scatter(res_Rayleigh_align, res_its, color='k',
            s=s_marker, alpha=alpha_marker, linewidths=linewidth_marker)
axits.set_ylabel("Intra-trial\nstability", linespacing=0.8)
axits.set_xlim(xmin=0)
axits.set_yticks([0,1])
axits.set_xticks([0,1])

axdim=axs['G']
axdim.scatter(res_Rayleigh_align[:,::-1], res_dim, color='k',

            s=s_marker, alpha=alpha_marker, linewidths=linewidth_marker)

axdim.axhline(dim_input, color='gray', ls='--')
axdim.set_ylabel("Dimension.")
axdim.set_ylim(ymin=0)
axdim.text(0.5, 10.5,"Input", fontsize=7)
axdim.set_xticks([0,1])

axalign=axs['H']
axalign.scatter(res_Rayleigh_align, res_align_spont, color='k',
            s=s_marker, alpha=alpha_marker, linewidths=linewidth_marker)

axalign.set_ylabel("Alignment with\n spont. activity", linespacing=0.8)
axalign.set_xlim(xmin=0)
axalign.set_yticks([0,0.2])
axalign.set_xticks([0,1])

axits.set_xlabel("Feedforward - Recurrent Alignment", linespacing=0.8)
axits.xaxis.set_label_coords(0.5, -0.3)

axalign.set_xlabel("FF-Rec.\nAlignment", linespacing=0.8)
axalign.xaxis.set_label_coords(0.5, -0.15)

# for ax in ["F", "H"]:
#     axs[ax].xaxis.set_label_coords(0.5, -0.15)

for ax in ["F", "G", "H", "E"]:
    axs[ax].yaxis.set_label_coords(-0.3, 0.5)

for ax in ["A", "B", "X", "D", "C"]:
    axs[ax].yaxis.set_label_coords(-0.32, 0.5)



for ax in [ "X", "D", "C"]:
    axs[ax].set_yticks([0, 0.4])
    axs[ax].yaxis.set_label_coords(-0.32, 0.45)

for ax in [*axs]:
    axs[ax].tick_params(length=1.5, width=0.5, labelsize=7)


fig.savefig("pictures/summary_4bf.pdf", dpi=200)




