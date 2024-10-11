#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:30:43 2022

@author: sigridtragenap
"""


import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


from tools.networks_definition import LinearRecurrentNetwork, create_W
import Analysis.Alignment_funs as AlgnFunc
import tools.helper_funs as hf
from tools.plotting_funcs import cm2inch, color_al, color_r

from matplotlib import cm

viridis = cm.get_cmap('viridis')
mpl.rcParams['figure.dpi'] = 200


#%%
seed_net=184


n_neuron=200
n_stim=8
n_trial=50
n_events=n_trial*n_stim
run_stab=True

simulation_params={
    'run_stab': run_stab,
    'n_neuron': n_neuron,
    'R': 0.85,
    'R_l4': 0.85,
    'sigma_spont': 1.5,
    'tuning_strength': 1,
    'sigma_noise_evoked': 1.5,
    'sigma_time_evoked': 0.2,
    'n_dim_spont': 20,
    'n_trial' :n_trial ,
    'n_stim' : n_stim,
    'n_dim_population': 10,
    #'n_events_stab_intv' : 10,
    }

##params simulation
n_drives=300

#intra und inter
dt=0.1
t1=100
t2=120
idx1=int(t1/dt)
idx2=int(t2/dt)
delta_dist=idx2-idx1
noise_level_time=0.25
T=120 #full duration stochastic simulation
sigma_ttc=0.02


#Set up network
w_rec=create_W(n_neuron, mean_connect=0,
                     symmetric=True, var_conn=1,
				    spectral_bound=simulation_params["R"],
                    seed=seed_net)
eigvals_net, eigvecs_net = linalg.eigh(w_rec)
reccurent_network = LinearRecurrentNetwork(w_rec)


#creating 'spont' ensemble
C_in_spont= hf.create_n_dim_gauss(
    simulation_params['n_dim_spont'],
    base_patterns=eigvecs_net[:,::-1])
C_sp_output = reccurent_network.transform_sigma_input(C_in_spont)



list_eigenvectors=np.asarray([0,5,100])
Stimensembles =[]
x=np.arange(n_neuron)[::-1]
eigbasis = reccurent_network.eigvecs
for idx in list_eigenvectors:
    indices = np.roll(x, -1*idx)
    C_input = hf.create_n_dim_gauss(simulation_params['n_dim_population'], base_patterns=eigbasis[:,indices],
                                    n_neurons=n_neuron)
    stims = hf.sample_from_normal(C_input,
                                  n_samples=n_drives)
    norms=linalg.norm(stims, axis=1)
    stims /= norms[:,None]
    Stimensembles.append(stims)
Nensemble = len(Stimensembles)

#%%


res_ttc = []
res_intra=[]
res_spont_align=[]
res_pattern_align=[]


for i in range(Nensemble):
    if i%10==0: print(i)
    ndrive=len(Stimensembles[i])
    res=np.empty(ndrive)*np.nan

    drive_resp= reccurent_network.res_Input_mat(Stimensembles[i].T).T
    norms=linalg.norm(drive_resp, axis=1)
    drive_resp /= norms[:,None]

    #calc intratrial stability
    for ipat in range(ndrive):
        all_resp = reccurent_network.time_dep_input_dW(Stimensembles[i][ipat],
                                                            dt=dt, T=T,
                                                            sigma=noise_level_time)

        C=np.corrcoef(all_resp)

        res[ipat]=np.mean(np.diag(C, k=delta_dist))

    res_intra.append(res)
    #print(res)

    #ttc
    res=np.empty(ndrive)*np.nan
    for ipat in range(ndrive):
        all_trials = hf.sample_from_normal(np.eye(n_neuron)*sigma_ttc,
                                           mean=Stimensembles[i][ipat],
                                           n_samples=100)
        responses = reccurent_network.res_Input_mat(all_trials.T).T
        C=np.corrcoef(responses)

        # plt.imshow(C,cmap='RdBu_r', vmin=-1, vmax=1)
        # plt.show()
        res[ipat]=np.nanmean(C[np.triu_indices(100, k=1)])
    res_ttc.append(res)

    res_pattern_align.append(
        np.diag(
            drive_resp@C_sp_output@drive_resp.T)/np.trace(C_sp_output))

    res_spont_align.append(np.diag(
        Stimensembles[i]@C_sp_output@Stimensembles[i].T)/np.trace(C_sp_output))




#%% plot first eigenvector

idx_popul = [0,] #maximum aligned EV


fig, (ax) = plt.subplots(figsize=cm2inch(2.5, 3), nrows=1,
                            sharex=True)
plt.subplots_adjust(bottom=0.25, top=0.9, left=0.25, right=0.92
                   )

colors_data=[plt.cm.viridis(0.6),
                plt.cm.viridis(0.9),
                plt.cm.viridis(0.25),][::-1]

labels = ["Maximal", "Intermediate", "High", ]

for idx, pop in enumerate(idx_popul):

    plt.scatter(res_pattern_align[pop], res_intra[pop], s=1,
                color=colors_data[idx], edgecolor='none', alpha=0.6,
         vmin=-0.2,vmax=1.5, zorder=10-i)


    #Korrelationslinie einzeichnen
    x_data = res_pattern_align[pop]
    y_data = res_intra[pop]
    coef = np.polyfit(x_data, y_data,1); poly1d_fn = np.poly1d(coef)
    xborder=0.00
    x_plot=[np.min(x_data)-xborder,np.max(x_data)+xborder]
    plt.plot(x_plot, poly1d_fn(x_plot), c=colors_data[idx],linewidth=0.8,
             zorder=20-i,
             label=labels[idx])

plt.xlabel("Alignment with\nspont. activity", fontsize=7,
           linespacing=0.8)
plt.ylabel("Intra-trial\nstability", fontsize=7,
           linespacing=0.8)

ax.set_yticks([0.0,  1]);
ax.set_xticks([0.0,  0.2]);

plt.xlim([0,None])
plt.ylim([0,1]); plt.yticks([0,1])
plt.xticks([0,0.2], )
ax=plt.gca()
#ax.tick_params(length=1, width=0.5)
ax.tick_params(length=1, width=0.5, labelsize=7)
ax.yaxis.set_label_coords(-0.1, 0.5, )
ax.xaxis.set_label_coords(0.5,-0.15 )
#plt.tight_layout()
fig.savefig("pictures/F8g_Alignmentspont_its.pdf", dpi=200)



#%% Repeat for trial trial correlation

#three pops
idx_popul = [0,]


fig, (ax) = plt.subplots(figsize=cm2inch(2.5, 3), nrows=1,
                            sharex=True)
plt.subplots_adjust(bottom=0.25, top=0.9, left=0.25, right=0.92
                   )
colors_data=[plt.cm.viridis(0.9),
                plt.cm.viridis(0.6),
                plt.cm.viridis(0.25),][::-1]

labels = ["Maximal", "High", "Intermediate"]
#["Pop. 1", "Pop. 12", "Pop. 3"]

for idx, pop in enumerate(idx_popul):

    plt.scatter(res_pattern_align[pop], res_ttc[pop], s=1,
                color=colors_data[idx], edgecolor='none', alpha=0.6,
         vmin=-0.2,vmax=1.5, zorder=10-i)


    #Korrelationslinie einzeichnen
    x_data = res_pattern_align[pop]
    y_data = res_ttc[pop]
    coef = np.polyfit(x_data, y_data,1); poly1d_fn = np.poly1d(coef)
    xborder=0.005
    x_plot=[np.min(x_data)-xborder,np.max(x_data)+xborder]
    plt.plot(x_plot, poly1d_fn(x_plot), c=colors_data[idx],linewidth=0.8,
             zorder=20-i,
             label=labels[idx])

plt.xlabel("Alignment with\nspont. activity", fontsize=7,
           linespacing=0.8)
plt.ylabel("Trial-to-trial\ncorrelation", fontsize=7,
           linespacing=1)

# leg=plt.legend(title='FF-Rec Alignment\n\n\n', frameon=False,
#             handlelength=0.2, loc=(0.1,0.0), handletextpad=0.3,
#             labelspacing=0, ncol=1, columnspacing=0.2, borderpad=0.1
#             )
# for legobj in leg.legendHandles:
#     legobj.set_alpha(1)
#     legobj.set_linewidth(2.0)
    #legendHandles[0]._sizes = [30]


plt.xlim([-0.01,None])
plt.ylim([0,1]); plt.yticks([0,1])
plt.xticks([0,0.2], )
ax=plt.gca()
ax.tick_params(length=1, width=0.5, labelsize=7)
ax.yaxis.set_label_coords(-0.1, 0.5, )
ax.xaxis.set_label_coords(0.5,-0.15 )
#plt.tight_layout()
fig.savefig("pictures/FSI9_Modelprediction_ttc.pdf", dpi=200)

