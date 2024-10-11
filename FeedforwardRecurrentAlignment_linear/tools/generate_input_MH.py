#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:44:13 2022

@author: sigridtragenap
"""

import numpy as np
from scipy import linalg


def shuffle_frame(act_frame, index, nmaps=None):
    f_fft=np.fft.rfft2
    f_ifft=np.fft.irfft2

    rng = np.random.RandomState(index)

    fft=f_fft(act_frame, axes=(0,1))
    absfft=np.abs(fft)
    count=1 if nmaps is None else nmaps
    result=np.empty((count,)+act_frame.shape,dtype=act_frame.dtype)*np.nan
    for i in range(count):
        angles=np.angle(fft)
        rng.shuffle(angles.flat)
        fft_shuffled=absfft*np.exp(1j*angles)
        shuffled=f_ifft(fft_shuffled, s=act_frame.shape, axes=(0,1))
        result[i]=shuffled
    return result[0] if nmaps is None else result


def get_additive_noise(full_shape, network_params, index, noise_level=None):
    #constant input in time!
    input_shape_woneuron = full_shape[:-2]
    N,M = full_shape[-2], full_shape[-1]
    nevents = np.prod(input_shape_woneuron)

    if noise_level is None:
        ## Input parameters
        sig1 = network_params['sigma_x_input']
        sig2 = 2*sig1
    else:
        sig1 = noise_level
        sig2 = 2*sig1

    #generate random input
    rng_start_activity = np.random.default_rng(index)
    input_rnd = rng_start_activity.random(size=[nevents, N, M]) #UNiform

    if np.allclose(0,sig1):
        additive_noise = np.copy(input_rnd)
    else:
        #apply convolution
        #use convolution with MH to get spatial scale in noisy input
        #define convolutio kernels
        x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
        kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
        kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
        input_smo = np.real(np.fft.ifft2(np.fft.fft2(kern1-kern2)[None,:,:]*np.fft.fft2(input_rnd,axes=(1,2)), axes=(1,2)))
        additive_noise = input_smo
    #reshape to fit original specifified shape
    #print(input_shape_woneuron,N,M, nevents, additive_noise.shape)
    additive_noise = additive_noise.reshape(nevents,N*M)
    additive_noise = (additive_noise - (additive_noise.mean(-1)[:,None]))/additive_noise.std(-1)[:,None]
    additive_noise = additive_noise.reshape(input_shape_woneuron+(N*M,))
    return additive_noise
