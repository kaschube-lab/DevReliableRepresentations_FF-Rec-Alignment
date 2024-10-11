#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:20:13 2022

@author: sigridtragenap
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy import linalg
import tools.helper_funs as hf


def get_Alignment_all(network, inputs):
    input_pats = inputs['evoked_input_direct']
    input_pats /= np.linalg.norm(inputs['evoked_input_direct'], axis=1)[:,None]
    alignment = hf.ev_explained_variance(network.W_connect, input_pats.T)
    return np.nanmean(alignment)

def get_Alignment_first(network, inputs):
    input_pats = inputs['evoked_input_direct']
    pca=PCA(n_components=2)
    pca=pca.fit(input_pats)
    comp1=pca.components_[0]
    alignment = comp1@network.W_connect@comp1
    return alignment

def patternsA_ExplainVariance_ofB(framesA, framesB):
    framesB_mean =  framesB - np.nanmean(framesB, axis=0)[None,:]
    norm = np.linalg.norm(framesB, axis=1)
    framesB_cov=np.cov(framesB_mean.T)

    framesA_mean =  framesA - np.nanmean(framesA, axis=0)[None,:]
    norm = np.linalg.norm(framesA_mean, axis=1)
    framesA_white = framesA_mean/norm[:,None]

    #A in B
    projections = np.dot(framesB_mean, framesA_white.T)
    varratios=np.var(projections, axis=0)/np.trace(framesB_cov)
    return varratios

def plot_PC_spectrum(patterns, fig=None, c=plt.cm.plasma(0.5)):
    if fig is None:
        fig=plt.figure(figsize=(2, 1.5))
    assert len(patterns.shape)==2
    #calc PCA
    pca=PCA()
    pca =pca.fit(patterns)
    varratios=pca.explained_variance_ratio_

    plt.plot(np.arange(len(varratios))+1,varratios, c=c)
    plt.xlim(xmax=20, xmin=0)
    plt.xlabel("PC Index")
    plt.ylabel("Fraction variance expl.")
    return fig, varratios


#ToDO
def plot_overlap_A_in_B(patterns_a, patterns_b, fig=None):
    if fig is None:
        fig=plt.figure(figsize=(2, 1.5))
    assert len(patterns_a.shape)==2  #shape: num_patterns, neurons,
    assert len(patterns_b.shape)==2

    #mean correction
    patterns_a=patterns_a - np.nanmean(patterns_a, axis=0)[None,:]
    patterns_b=patterns_b - np.nanmean(patterns_b, axis=0)[None,:]

    pca_b=PCA()
    pca_b =pca_b.fit(patterns_b)  #
    components_b = pca_b.components_
    #print(components_b.shape, patterns_a.shape)
    var_A_in_compB = patternsA_ExplainVariance_ofB(components_b, patterns_a)
    varratios=pca_b.explained_variance_ratio_

    plt.plot(np.arange(len(varratios))+1,varratios, c=plt.cm.plasma(0.5),
             label="spont.")
    plt.plot(np.arange(len(var_A_in_compB))+1,var_A_in_compB,
          c=plt.cm.viridis(0.35),label="evoked")
    plt.xlim(xmax=20, xmin=0)
    plt.xlabel("PC Index")
    plt.ylabel("Fraction variance expl.")
    return fig, varratios, var_A_in_compB

def w_statistics(W):
    norm = np.sqrt(np.sum(np.square(W)))
    eigvals = linalg.eigvals(W)
    spectral_rad = np.max(eigvals[1:].real)
    pop_EV = eigvals[0].real
    res={
      "norm": norm,
      "R": spectral_rad,
      "max_EV": pop_EV}
    return res


