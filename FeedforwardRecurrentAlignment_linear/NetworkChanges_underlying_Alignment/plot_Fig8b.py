#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:38:27 2023

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
from tools.plotting_funcs import cm2inch

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["figure.figsize"]=[2.5, 2.5]

key_file = "results_simulation/model_{}Change{}_Optimal.npz"
n_sim_total=5

#%%



color_sp= plt.cm.plasma(0.5)
color_ev= plt.cm.viridis(0.35)

figsize_narrow = (2, 2.8)
figsize_curves = (3, 2.8)



n_sim_total=5
Nanimals=n_sim_total

markersize=5
markeralpha=0.5
markerlw=0.5

#%%
cond_list = ['FFOpt', 'RecOpt', 'BothOpt']


xticklabels =['Naive', 'Aligned']

#%% Data in one dict

n_sim_total=5
n_neuron=200


#sigma_plot_spread=0.05
#scatter_animals=sigma_plot_spread*np.random.randn(n_sim_total)


res_dicts_readout={}
for cond in cond_list:


    res_ff_rec_alignment_first=[]
    res_ff_rec_alignment_samples=[]

    res_Dimensionality_ev_in=[]
    res_Dimensionality_ev_out=[]

    res_spont_alignment_mean=[]



    for i in tqdm(range(n_sim_total)):
        #print(i)
        data_naive = np.load(key_file.format(
            cond,i),
            allow_pickle=True)['data_naive'].item()

        data_learned = np.load(key_file.format(
            cond,i),
            allow_pickle=True)['data_exp'].item()


        spE = data_learned['spont_patterns']
        evE = data_learned['evoked_patterns']
        evE_flat = data_learned['evoked_patterns'].reshape(
            -1,n_neuron)

        spN = data_naive['spont_patterns']
        evN = data_naive['evoked_patterns']
        evN_flat = data_naive['evoked_patterns'].reshape(
            -1,n_neuron)


        res_Dimensionality_ev_in.append(np.asarray([
            hf.dimensionality(linalg.eigvalsh(data_naive['evoked_cov_in'])),
            hf.dimensionality(linalg.eigvalsh(data_learned['evoked_cov_in']))]))

        res_Dimensionality_ev_out.append(np.asarray([
            hf.dimensionality(linalg.eigvalsh(data_naive['evoked_cov_out'])),
            hf.dimensionality(linalg.eigvalsh(data_learned['evoked_cov_out']))]))


        #hf.trial_to_trial_correlation(evN, 16, 15)

        res_spont_alignment_mean.append(np.asarray([
            AlgnFunc.patternsA_ExplainVariance_ofB(evN_flat, spN).mean(),
            AlgnFunc.patternsA_ExplainVariance_ofB(evE_flat, spE).mean()
            ]))


        res_ff_rec_alignment_first.append(np.asarray(
            np.load(key_file.format(
                cond,i),allow_pickle=True)['align_first']
            ))
        res_ff_rec_alignment_samples.append(np.asarray(
            np.load(key_file.format(
                cond,i),allow_pickle=True)['align_all']
            ))

    dict_all_variables = dict(
        res_Dimensionality_ev_out=res_Dimensionality_ev_out,
        res_Dimensionality_ev_in=res_Dimensionality_ev_in,
        res_spont_alignment_mean=res_spont_alignment_mean,
        res_ff_rec_alignment_samples=res_ff_rec_alignment_samples,
        res_ff_rec_alignment_first=res_ff_rec_alignment_first,

        )
        # res_evoked_inearly_evoked=res_evoked_inearly_evoked,
        # res_evoked_inearly_spont=res_evoked_inearly_spont,
        # res_spont_inearly_evoked=res_spont_inearly_evoked,
        # res_spont_inearly_spont=res_spont_inearly_spont)

    res_dicts_readout.update({cond: dict_all_variables})



#%% Validation plots


##Alignment
key_plot='res_spont_alignment_mean'
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):

    data_plot=np.asarray(res_dicts_readout[cond][key_plot]).T
    ax.plot(data_plot, color='k', ls='', marker='o', ms=3, alpha=0.3)
    ax.errorbar(x=[0,1],    y=data_plot.mean(1),
                yerr=stats.sem(data_plot, axis=1),
                color='k',  marker='s', ls='', ms=5, alpha=1)

axs[0].set_ylabel("Alignment with\nspont. activity",
           linespacing=1)
axs[1].set_xlabel("Network state",
           linespacing=1)
plt.ylim(ymin=0,ymax=0.3)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.xticks([0,1], xticklabels)
plt.show()
fig.savefig("pictures/Alignment_spont.pdf")


#Alignment (samples)
key_plot='res_ff_rec_alignment_samples'
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):

    data_plot=np.asarray(res_dicts_readout[cond][key_plot]).T
    ax.plot(data_plot, color='k', ls='', marker='o', ms=3, alpha=0.3)
    ax.errorbar(x=[0,1],    y=data_plot.mean(1),
                yerr=stats.sem(data_plot, axis=1),
                color='k',  marker='s', ls='', ms=5, alpha=1)

axs[0].set_ylabel("Alignment (samples)",
           linespacing=1)
axs[1].set_xlabel("Network state",
           linespacing=1)
plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.xticks([0,1], xticklabels)
plt.show()
fig.savefig("pictures/Alignment_spont.pdf")


#FF-Rec Alignment of first PC
key_plot='res_ff_rec_alignment_first'
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)
for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):
    data_plot=np.asarray(res_dicts_readout[cond][key_plot]).T
    ax.plot(data_plot, color='k', ls='', marker='o', ms=3, alpha=0.3)
    ax.errorbar(x=[0,1],    y=data_plot.mean(1),
                yerr=stats.sem(data_plot, axis=1),
                color='k',  marker='s', ls='', ms=5, alpha=1)
axs[0].set_ylabel("Alignment (PC)",
           linespacing=1)
axs[1].set_xlabel("Network state",
           linespacing=1)
#plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.xticks([0,1], xticklabels)
plt.show()
fig.savefig("pictures/Alignment_spont.pdf")

#Dimensionality
key_plot='res_Dimensionality_ev_out'
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)
for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):

    data_plot=np.asarray(res_dicts_readout[cond][key_plot]).T
    #ax.scatter( [0,1]*n_sim_total, data_plot)

    ax.plot(data_plot, color='k', ls='', marker='o', ms=3, alpha=0.3)
    ax.errorbar(x=[0,1],    y=data_plot.mean(1),
                yerr=stats.sem(data_plot, axis=1),
                color='k',  marker='s', ls='', ms=5, alpha=1)

axs[0].set_ylabel("Dimensionality",
           linespacing=1)
axs[1].set_xlabel("Network state",
           linespacing=1)
#plt.ylim(ymin=0,ymax=6)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.xticks([0,1], xticklabels)
ax.yaxis.set_label_coords(-0.17, 0.5, )
plt.show()
fig.savefig("pictures/Dim_out.pdf")



#%% Calc Variance explained
Nproj=10
def get_explained_A_inB(dataA,dataB, Nproj=10):

    _, _, ev_var_e = AlgnFunc.plot_overlap_A_in_B(dataA,
                                 dataB)
    plt.close()
    res_explained= ev_var_e.cumsum()
    return res_explained[Nproj]

def get_explvar_A_B_split(dataA,dataB,Nproj=10):
    n_event_half = dataA.shape[0]//2
    dataA = dataA[:n_event_half]
    dataB = dataB[n_event_half:]
    return get_explained_A_inB(dataA,dataB, Nproj)


#%% Data in one dict

n_sim_total=5
n_neuron=200
sigma_plot_spread=0.05
scatter_animals=sigma_plot_spread*np.random.randn(n_sim_total)


res_dicts={}
for cond in cond_list:


    res_evoked_control=[]
    res_spont_control=[]

    res_evoked_inearly_evoked=[]
    res_evoked_inearly_spont=[]
    res_evoked_inearly_all=[]


    res_spont_inearly_evoked=[]
    res_spont_inearly_spont=[]
    res_spont_inearly_all=[]


    for i in tqdm(range(n_sim_total)):
        #print(i)
        data_naive = np.load(key_file.format(
            cond,i),
            allow_pickle=True)['data_naive'].item()

        data_learned = np.load(key_file.format(
            cond,i),
            allow_pickle=True)['data_exp'].item()


        spE = data_learned['spont_patterns']
        evE = data_learned['evoked_patterns'].reshape(
            -1,n_neuron)

        spN = data_naive['spont_patterns']
        evN = data_naive['evoked_patterns'].reshape(
            -1,n_neuron)


        res_evoked_control.append(get_explvar_A_B_split(evE,evE,Nproj))
        res_spont_control.append(get_explvar_A_B_split(spE,spE,Nproj))

        res_evoked_inearly_evoked.append(
            get_explvar_A_B_split(evE, evN))

        res_evoked_inearly_spont.append(
            get_explvar_A_B_split(evE, spN))

        pool_all = np.concatenate([evN,spN])
        np.random.shuffle(pool_all)
        res_evoked_inearly_all.append(
            get_explvar_A_B_split(evE, pool_all))


        res_spont_inearly_evoked.append(
            get_explvar_A_B_split(spE, evN))

        res_spont_inearly_spont.append(
            get_explvar_A_B_split(spE, spN))

        res_spont_inearly_all.append(
            get_explvar_A_B_split(spE, pool_all))

    dict_all_variables = dict(
        res_evoked_control=res_evoked_control,
        res_spont_control=res_spont_control,
        res_evoked_inearly_evoked=res_evoked_inearly_evoked,
        res_evoked_inearly_spont=res_evoked_inearly_spont,
        res_spont_inearly_evoked=res_spont_inearly_evoked,
        res_spont_inearly_spont=res_spont_inearly_spont)

    res_dicts.update({cond: dict_all_variables})

#%% plot
label_list = ["Naïve\nSpont.",  "Naïve\nEvoked","\n"]
color_source_list=[ plt.cm.magma(0.5),plt.cm.viridis(0.35),  'k']

## plot predict evoked
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):
    cond_list_plot = [
                  "res_evoked_inearly_spont",
                  "res_evoked_inearly_evoked",]
    for icond in range(len(cond_list_plot)):

        data_condition=np.asarray(res_dicts[cond][cond_list_plot[icond]])
        color=color_source_list[icond]
        x_plot_data = np.ones(Nanimals)*(icond)+scatter_animals
        axs[idx_ax].scatter(x_plot_data, data_condition,
                    edgecolors=color, facecolors='none',
                    s=markersize, alpha=markeralpha,                    linewidth=markerlw)

        axs[idx_ax].errorbar(icond, data_condition.mean(), yerr=stats.sem(data_condition),
                    marker="s", ms=3.5, color=color,
                          lw=1 , zorder=10, capsize=3, capthick=0.8,)


axs[0].set_ylabel("Experienced evoked\nvariance explained by",
           linespacing=1)
plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.yticks([0,0.5,1])
ax.yaxis.set_label_coords(-0.17, 0.5, )
for ax in axs:
    ax.tick_params(axis='x', pad=4)
plt.xticks(np.arange(len(label_list))[:2], label_list[:2])

plt.show()
fig.savefig("pictures/Model_predict_evoked_combined.pdf", dpi=200)


## plot predict spontaneous
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):

    cond_list_plot = ["res_spont_inearly_spont",
                  "res_spont_inearly_evoked",]
    for icond in range(len(cond_list_plot)):

        data_condition=np.asarray(res_dicts[cond][cond_list_plot[icond]])
        color=color_source_list[icond]
        x_plot_data = np.ones(Nanimals)*(icond)+scatter_animals
        axs[idx_ax].scatter(x_plot_data, data_condition,
                    edgecolors=color, facecolors='none',
                    s=markersize, alpha=markeralpha,
                    linewidth=markerlw)

        axs[idx_ax].errorbar(icond, data_condition.mean(), yerr=stats.sem(data_condition),
                    marker="s", ms=3.5, color=color,
                          lw=1 , zorder=10, capsize=3, capthick=0.8,)

axs[0].set_ylabel("Experienced spont.\nvariance explained by",
           linespacing=1)
plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.yticks([0,0.5,1])
ax.yaxis.set_label_coords(-0.17, 0.5, )
for ax in axs:
    ax.tick_params(axis='x', pad=4)
plt.xticks(np.arange(len(label_list))[:2], label_list[:2])
plt.show()
fig.savefig("pictures/Model_predict_spont_combined.pdf", dpi=200)



#%% #Same plots as above but without labels

label_list = ["Naïve\nSpont.",  "Naïve\nEvoked","\n"]
color_source_list=[ plt.cm.magma(0.5),plt.cm.viridis(0.35),  'k']

#predict evoked
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):
    cond_list_plot = [
                  "res_evoked_inearly_spont",
                  "res_evoked_inearly_evoked",]
    for icond in range(len(cond_list_plot)):

        data_condition=np.asarray(res_dicts[cond][cond_list_plot[icond]])
        color=color_source_list[icond]
        x_plot_data = np.ones(Nanimals)*(icond)+scatter_animals
        axs[idx_ax].scatter(x_plot_data, data_condition,
                    edgecolors=color, facecolors='none',
                    s=markersize, alpha=markeralpha,
                    linewidth=markerlw)

        axs[idx_ax].errorbar(icond, data_condition.mean(), yerr=stats.sem(data_condition),
                    marker="s", ms=3.5, color=color,
                          lw=1 , zorder=10, capsize=3, capthick=0.8,)
plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.yticks([0,0.5,1])


ax.yaxis.set_label_coords(-0.17, 0.5, )
plt.xticks([])
plt.show()
fig.savefig("pictures/Model_predict_evoked_combined_v2.pdf", dpi=200)


#predict spontaneous
fig, (axs)=plt.subplots(figsize=cm2inch(8,3), ncols=3,
                                  sharex=True, sharey=True)
plt.subplots_adjust(left=0.25, right=0.92, bottom=0.25,  top=0.95,
                    wspace=0.5)

for idx_ax,(cond, ax) in enumerate(zip(cond_list, axs)):
    cond_list_plot = ["res_spont_inearly_spont",
                  "res_spont_inearly_evoked",]
    for icond in range(len(cond_list_plot)):

        data_condition=np.asarray(res_dicts[cond][cond_list_plot[icond]])
        color=color_source_list[icond]
        x_plot_data = np.ones(Nanimals)*(icond)+scatter_animals
        axs[idx_ax].scatter(x_plot_data, data_condition,
                    edgecolors=color, facecolors='none',
                    s=markersize, alpha=markeralpha,
                    linewidth=markerlw)

        axs[idx_ax].errorbar(icond, data_condition.mean(), yerr=stats.sem(data_condition),
                    marker="s", ms=3.5, color=color,
                          lw=1 , zorder=10, capsize=3, capthick=0.8,)
plt.ylim(ymin=0,ymax=1)
plt.xlim(xmin=-0.2, xmax=1.1)
plt.yticks([0,0.5,1])
ax.yaxis.set_label_coords(-0.17, 0.5, )
plt.xticks([])
plt.show()
fig.savefig("pictures/Model_predict_spont_combined_v2.pdf", dpi=200)

