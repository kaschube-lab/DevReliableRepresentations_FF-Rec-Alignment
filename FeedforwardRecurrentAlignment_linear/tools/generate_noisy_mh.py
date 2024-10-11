#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:53:00 2022

@author: sigridtragenap
"""

import numpy as np
import scipy.linalg as la

def get_kmax_eigvals(w_rec,network_params):
    ## get estimate of spatial scale of pattern by looking at peak in spectrum of M
    N,M = w_rec.shape[0], w_rec.shape[1]
    w_rec0 = abs(np.fft.fftshift(np.fft.fft2(w_rec[N//2,M//2,:,:])))
    kmax_flat = np.argmax(w_rec0)
    kmax = np.array([abs(kmax_flat%N-N//2),abs(kmax_flat//N-M//2)])
    w_rec = w_rec.reshape(N*M,N*M)

    ## normalize M such that real part of maximal eigenvalue is given by network_params['nonlin_fac']
    all_eigenvals =la.eigvals(w_rec)
    max_eigenval = np.nanmax(np.real(all_eigenvals))
    w_rec = network_params['nonlin_fac']*w_rec/np.real(max_eigenval)
    return w_rec, kmax, all_eigenvals*network_params['nonlin_fac']/np.real(max_eigenval)


def gauss(delta,inh_factor):
    return 1./inh_factor**2*np.exp(-delta/2./inh_factor**2)

def convolve_with_MH(input_rnd,N,M,sigma1=2,sigma2=6,return_real=True,padd=0):
    ''' use convolution with MH to get spatial scale in noisy input'''
    h,w = input_rnd.shape
    x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N+2*padd),np.linspace(-M//2+1,M//2,M+2*padd))
    sig1 = sigma1#2
    sig2 = sigma2#3*sig1
    kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
    kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
    diff_gauss = kern1-kern2
    to_smooth = np.pad(input_rnd,padd,'constant')
    input_smo = np.fft.ifft2(np.fft.fft2(diff_gauss)*np.fft.fft2(to_smooth,axes=(0,1)), axes=(0,1))
    input_smo = np.fft.fftshift(input_smo)
    hn,wn = input_smo.shape
    if padd>0:
        input_smo = input_smo[hn//2-h//2:hn//2+h//2,wn//2-w//2:wn//2+w//2]
    if return_real:
        return np.real(input_smo)
    else:
        return input_smo


def noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,orientation_sd,\
a1,inh_factor,pbc=True,index=4876,full_output=True,rotate_by=None,conv_params=None):
    coord_x,coord_y= np.meshgrid(np.arange(N),np.arange(M)) #N:x dimension; M:y dim.

    new_index = index
    np.random.seed(index)


    ## periodic boundary conditions
    if pbc:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]
        absdeltax = np.abs(deltax)
        absdeltay = np.abs(deltay)
        idxx = np.argmin([absdeltax, N-absdeltax],axis=0)
        idxy = np.argmin([absdeltay, M-absdeltay],axis=0)

        deltax = deltax*(1-idxx) + np.sign(deltax)*(absdeltax-N)*idxx
        deltay = deltay*(1-idxy) + np.sign(deltay)*(absdeltay-M)*idxy
    else:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]

    if mode=='short_range':
        if 'None' in noise_type:
            '''homogeneous mhs'''
            sigmax=conv_params['s1_x']
            sigmay=conv_params['s1_x']
            delta = (deltax)**2/sigmax**2 + (deltay)**2/sigmay**2

            mh1=gauss(delta,1.)/2./np.pi
            mh2=gauss(delta,inh_factor)/2./np.pi
            anisotropic_mh = ( mh1 - a1*mh2 )
        elif noise_type=='postsyn':
            ## generate spatial scale in x direction
            if conv_params['do_convolution_x']:
                sigmax_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_x'],sigma2=conv_params['s2_x'])[:,:,None,None]
            #elif conv_params['do_lowpass_x']:
            #    sigmax_noise = smm.low_normalize(np.random.randn(M,N), mask=None, sigma=3)[:,:,None,None]
            else:
                sigmax_noise = np.random.randn(M,N)[:,:,None,None]
            sigmax_noise = (sigmax + sigmax_noise/np.std(sigmax_noise)*sigmax_sd)
            sigmax_noise[sigmax_noise<0]=0.0

            ## generate eccentricity array
            if conv_params['do_convolution_ecc']:
                ecc_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_ecc'],sigma2=conv_params['s2_ecc'])[:,:,None,None]
            #elif conv_params['do_lowpass_ecc']:
            #    ecc_noise = smm.low_normalize(np.random.randn(M,N), mask=None, sigma=3)[:,:,None,None]
            else:
                ecc_noise = np.random.randn(M,N)[:,:,None,None]
            ecc_noise = ecc + ecc_noise/np.std(ecc_noise)*ecc_sd

            #rewrite using clip
            ecc_noise[ecc_noise>0.95] = 0.95
            ecc_noise[ecc_noise<0.0] = 0.0

            ## calculate spatial scale in y-direction
            sigmay_noise = sigmax_noise*np.sqrt(1 - ecc_noise**2)

            ## generate array of orientations
            z_noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
            if conv_params['do_convolution_ori']:
                z_noise = convolve_with_MH(z_noise,N,M,sigma1=conv_params['s1_ori'],sigma2=conv_params['s2_ori'],return_real=False,padd=conv_params['padd'])
            #elif conv_params['do_lowpass_ori']:
            #    z_noise = smm.low_normalize(z_noise, mask=None, sigma=3)
            elif conv_params['const_ori']:
                z_noise = np.zeros((M,N),dtype='complex')
            orientation_noise = np.angle(z_noise)*0.5
            orientation_noise = orientation + orientation_noise

            orientation_noise = orientation_noise + np.pi*(orientation_noise<(np.pi/2))
            orientation_noise = orientation_noise - np.pi*(orientation_noise>(np.pi/2))

            cos_noise = np.cos(orientation_noise)[:,:,None,None]
            sin_noise = np.sin(orientation_noise)[:,:,None,None]

            if not full_output:
                return ecc_noise[:,:,0,0],orientation_noise,sigmax_noise[:,:,0,0],sigmay_noise[:,:,0,0]

            delta = (deltax*cos_noise - deltay*sin_noise)**2/sigmax_noise**2 + (deltay*cos_noise + sin_noise*deltax)**2/sigmay_noise**2
            mh1 = gauss(delta,1.)/sigmay_noise/sigmax_noise/2./np.pi
            mh2 = gauss(delta,inh_factor)/sigmay_noise/sigmax_noise/2./np.pi
            anisotropic_mh = ( mh1 - a1*mh2 )



        w_rec=np.real(anisotropic_mh)

        if conv_params['do_Binomial_sampling']:
            rng=np.random.RandomState(seed=index)
            m1_sampled=rng.binomial(conv_params['K_E'], mh1)*conv_params['w_E']
            m2_sampled=rng.binomial(conv_params['K_I'], a1*mh2)*conv_params['w_I']
            w_rec=m1_sampled+m2_sampled

        return w_rec,new_index



def noisy_mh_wrap(conn_params, seed):
    rotate_by = None
    s1 = conn_params["sigmax"]
    s2 = conn_params["sigmax"]*conn_params["inh_factor"]
    conv_params = {'do_convolution_ori' : False,'padd' : 0,
        's1_ori' : s1,
        's2_ori' : s2,
        'do_convolution_x' : False,
        's1_x' : s1, 's2_x' : s2,
        'do_convolution_ecc' : False,
        's1_ecc' : s1, 's2_ecc' : s2,
        'do_lowpass_ori' : False,
        'do_lowpass_x' : False,
        'do_lowpass_ecc' : False,
         'const_ori' : False,
        'do_Binomial_sampling': False,}

    return noisy_mh(conn_params["N"],conn_params["M"],conn_params["mode"],
                    conn_params["noise_type"],conn_params["sigmax"],
                    conn_params["sigmax_sd"],conn_params["ecc"],
                    conn_params["ecc_sd"],conn_params["orientation"],\
                    conn_params["orientation_sd"],conn_params["amplitude"],
                    conn_params["inh_factor"],pbc=conn_params["pbc"],
                    index=seed,full_output=True,
    rotate_by=rotate_by,conv_params=conv_params)



def sampled_mh_wrap(N,M,mode,noise_type,sigmax,sigmax_sd,number_syn,ecc_sd,
                    orientation,orientation_sd,a1,inh_factor,pbc=True,
                    index=4876,full_output=True,version=0):
    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,
                       'do_Binomial_sampling': True, 'K_I': number_syn, 'K_E': number_syn, 'w_I':-1.*inh_factor , 'w_E': 1., }

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,number_syn,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,rotate_by=rotate_by,conv_params=conv_params)




if __name__=='__main__':
    import matplotlib.pyplot as plt

    ## Network size
    N=60
    M=60

    ## Connectivity settings
    ## sigmax and inh_factor define spatial scale
    sigmax = 1.8
    inh_factor = 2.        ## kappa in manuscript

    a1 = 1.                ## global strength of inhibition

    ## heterogeneity settings
    ecc = 0.8
    sigmax_sd = ecc*0.15/0.8
    ecc_sd = ecc*0.13
    orientation = 0.
    ori_sd = 1.0

    ## I used VERSION number for different network settings
    ## and index number for different heterogeneity settings
    VERSION = 1
    index = 1

    ''' plot different single mhs '''
    gauss_b,idx = noisy_mh_wrap(N,M,'short_range','postsyn',sigmax,sigmax_sd,ecc,\
    ecc_sd,orientation,ori_sd,a1,inh_factor,pbc=True,index=index,full_output=True,version=VERSION)

    ## show complete connectivity M
    plt.figure()#(figsize=(30,30))
    im=plt.imshow(gauss_b.reshape(N*M,N*M),interpolation='nearest',cmap='RdBu_r')#,vmax=0.05)
    plt.colorbar(im)

    ## show three examples of Mexican hats and one exemplary spectrum
    fig=plt.figure()
    ax=fig.add_subplot(141)
    show1 = np.fft.fftshift(gauss_b[2,3,:,:])
    im=ax.imshow(gauss_b[2,3,:,:],interpolation='nearest',cmap='RdBu_r')
    #plt.colorbar(im)
    ax=fig.add_subplot(142)
    ax.imshow(gauss_b[3,3,:,:],interpolation='nearest',cmap='RdBu_r')
    ax=fig.add_subplot(143)
    ax.imshow(np.fft.fftshift(gauss_b[2,5,:,:]),interpolation='nearest',cmap='RdBu_r')
    ax=fig.add_subplot(144)
    show3 = np.fft.fftshift(np.abs(np.fft.fft2(gauss_b[5,5,:,:])))
    ax.imshow(show3,interpolation='nearest',cmap='RdBu_r')
    plt.show()

