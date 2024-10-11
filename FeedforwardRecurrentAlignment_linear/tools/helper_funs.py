#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:30:52 2022

@author: sigridtragenap
"""

import numpy as np
from numpy.random import default_rng
from scipy import linalg

def dimensionality(eigvals):
    return np.sum(eigvals)**2/np.sum(eigvals**2)


def construct_mat(eigvals, eigvecs):
    #eigvecs.shape  neurons, number eigvecs
    assert len(eigvals)==eigvecs.shape[1]
    n_supply = eigvecs.shape[1]
    n_dim = eigvecs.shape[0]
    res_W = np.zeros((n_dim, n_dim))
    for i in range(n_supply):
        eigenvector = eigvecs[:,i]
        res_W += eigvals[i]* (eigenvector[:,None] @ eigenvector[None,:])
    return res_W

def construct_onb(n_dim, seed=None):
    #construct random, symmetric matrix
    rng = default_rng(seed=seed)
    A= rng.standard_normal(size=(n_dim, n_dim))
    A_sym = A + A.T
    assert np.allclose(A_sym, A_sym.T)
    #calculate eigenvectors
    #because A is symmetric, these will form an orthonormal basis
    _, eigvecs = linalg.eigh(A_sym)
    return eigvecs

def ev_explained_variance(sigma, eigvecs):
    #eigvecs.shape  neurons, number eigvecs
    return np.diag(eigvecs.T @ sigma @ eigvecs)

def construct_transformation_sigma(broadness=0.2, n_dim=200, ONB=None):
    if ONB is None:  #otherwise pass specific basis vectors
        ONB = construct_onb(n_dim)
    x=np.arange(n_dim)
    eigvals_broad = np.exp(-broadness*x)
    eigvals_broad *= n_dim/eigvals_broad.sum()
    #alternative: construct sigma, take sqrtm
    return construct_mat(np.sqrt(eigvals_broad), ONB)
    #take the real part because complex results are numerical errors
    #sigma onoy has positive eigenvalues per construction

def sample_from_normal(cov, mean=None, n_samples=1000, seed=None):
    if mean is None:
        mean = np.zeros(cov.shape[0])
    rng_state= np.random.default_rng(seed)
    sampled_vectors = rng_state.multivariate_normal(mean, cov, n_samples)
    return sampled_vectors

def create_n_dim_gauss(K, base_patterns=None, n_neurons=200):
    if base_patterns is None:
        base_patterns = construct_onb(n_neurons)
    eigvals_broad = np.exp(-np.arange(n_neurons)/(0.5*K))
    eigvals_broad *= n_neurons/eigvals_broad.sum()
    C_in = construct_mat(eigvals_broad, base_patterns)
    return C_in

def trial_to_trial_correlation(data_evoked, n_stim, n_ntrial):
    res=[]
    for istim in range(n_stim):
        res.append(np.nanmean(np.corrcoef(data_evoked[:,istim])[np.triu_indices(n_ntrial, k=1)]))
    return np.asarray(res)

def closest_matching_pats(patternsA, patternsB):
    n_a = patternsA.shape[0]
    C_all = np.corrcoef(patternsA, patternsB)[:n_a, n_a:]
    #C_all[np.diag_indices()]
    max_sim = np.max(C_all, axis=1)
    return max_sim

def selec_from_tuning(tuning, normkey=True):
    num_stim, num_cells = tuning.shape
    if num_stim==9:
        tuning = tuning[:8]
        return selec_from_tuning(tuning)
    if num_stim==17:
        tuning = tuning[:16]
        return selec_from_tuning(tuning)
    angles = np.arange(num_stim)*np.pi/float(num_stim)
    angles = np.exp(2j*angles)
    #norm=np.nansum(tuning, axis=0)
    norm=np.nansum(np.abs(tuning), axis=0)
    if normkey==False: norm=1
    opm = np.nansum(tuning*angles[:,None], axis=0)/norm
    #/norm
    return np.abs(opm)

def _np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def shuffled_signalcorr(dummy_TrialFrameNeur):
    num_frames, num_neur=dummy_TrialFrameNeur.shape
    if num_frames%2>0:
        return shuffled_signalcorr(dummy_TrialFrameNeur[:-1])
    shuffled_corr=[] = []
    N_half = num_frames//2
    for ineur in range(num_neur):
        D1 = dummy_TrialFrameNeur[:N_half,ineur]
        D2 = dummy_TrialFrameNeur[N_half:]
        res_12 = _np_pearson_cor(D1, D2)[0]

        del D1, D2

        D1 = dummy_TrialFrameNeur[N_half:,ineur]
        D2 = dummy_TrialFrameNeur[:N_half]

        res_21 = _np_pearson_cor(D1, D2)[0]
        shuffled_corr.append(0.5*(res_12+res_21))

        del D1, D2

    shuffled_corr = np.asarray(shuffled_corr)
    return shuffled_corr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N=100000

    res=[]
    for a in np.linspace(0.01, 0.5):
        x=np.arange(N)
        eigvals_broad = np.exp(-a*x)
        eigvals_broad *= N/eigvals_broad.sum()
        res.append(dimensionality(eigvals_broad))
    plt.plot(np.linspace(0.01, 0.5), res)
    plt.ylim(ymax=30)

    noise_transform_matrix = construct_transformation_sigma(
            broadness=0.3,
            n_dim=200,
        )

