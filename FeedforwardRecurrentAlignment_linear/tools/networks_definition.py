#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:56:16 2022

@author: sigridtragenap
"""
import numpy as np
from  matplotlib import colors
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.integrate as integrate



class LinearRecurrentNetwork:
    def __init__(self,
              W_connect ,
              tau_neur=1,
              sym=True):
        self.W_connect=W_connect
        self.Nneuron = self.W_connect.shape[0]
        #calc eigvals
        if sym:
            eigvals, eigvecs = linalg.eigh(self.W_connect)
        else:
            eigvals, eigvecs = linalg.eig(self.W_connect, left=True, right=False)
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.tau_r = tau_neur
        self.timescales=1-self.eigvals

        self.Interaction_mat =linalg.inv(np.eye(self.Nneuron)-self.W_connect)

    def overlap_x_ev(self,x):
        assert x.shape[-1]==self.eigvecs.shape[0]
        return np.dot(x, self.eigvecs)

    def steady_state(self,input_h):
        overlap_input = self.overlap_x_ev(input_h)
        prefac_ev= overlap_input/self.timescales
        act_ev= prefac_ev[None,:]*self.eigvecs
        return np.sum(act_ev, axis=-1)

    def res_Input_mat(self, input_M):
        assert input_M.shape[0]==self.Nneuron
        assert len(input_M.shape)==2
        return self.Interaction_mat @ input_M

    def time_resolved_sol(self, input_h, t, start_act=None):
        #still static input
        if start_act is None:
            start_act=np.zeros(self.Nneuron)
        overlap_start_a=self.overlap_x_ev(start_act)
        time_amplitude_start = np.exp(-t*self.timescales/self.tau_r)
        prefac_start = overlap_start_a*time_amplitude_start

        overlap_input = self.overlap_x_ev(input_h)
        time_amplitude_input= 1- np.exp(-t*self.timescales/self.tau_r)
        prefac_input = overlap_input*time_amplitude_input/self.timescales
        act_ev=(prefac_input+prefac_start)[None,:]*self.eigvecs
        return np.sum(act_ev, axis=-1)

    def sol_time_dependentinput(self, input_h_history,
                                  dt=0.1, start_act=None):
        if start_act is None:
            start_act=np.zeros(self.Nneuron)

        assert input_h_history.shape[-1]==len(self.eigvecs)
        Nsteps=input_h_history.shape[0]
        times = np.arange(0,Nsteps)*dt
        t=times[-1]

        #decay of initial
        overlap_startA=self.overlap_x_ev(start_act)
        time_amplitude_start = np.exp(-t*self.timescales/self.tau_r)
        prefac_start = overlap_startA*time_amplitude_start

        #integral
        overlap_inputs = self.overlap_x_ev(input_h_history)
        exponent=np.asarray((times[:,None]-t)*self.timescales[None,:]/self.tau_r,
                      dtype=np.float128)
        kernel_integral=np.exp(exponent)
        y_integrate= kernel_integral*overlap_inputs
        res_integral = integrate.simpson(y=y_integrate, dx=dt, axis=0)


        #time_amplitude_input=np.exp(-t*self.timescales/self.tau_r)/self.tau_r
        time_amplitude_input=1  #factor -t in integral to prevent overflow
        prefac_input = time_amplitude_input*res_integral
        act_ev=(prefac_input+prefac_start)[None,:]*self.eigvecs

        return np.sum(act_ev, axis=-1)

    def sol_Input_depT(self, input_h_history,
                     dt=0.1, start_act=None,
                     Nsteps=None):
        if start_act is None:
            start_act=np.zeros(self.Nneuron)

        assert input_h_history.shape[-1]==len(self.eigvecs)
        if input_h_history.shape[0]>1:
            Nsteps=input_h_history.shape[0]
        else:
            Nsteps=int(1e3)

        res=[]
        print(Nsteps)
        for istep in range(Nsteps-1):
            act=self.sol_time_dependentinput(input_h_history[:istep+1],
                                           dt=dt, start_act=start_act)
            res.append(act)
        return np.asarray(res)

    def time_dep_input_dW(self, input_h_mean,
                       dt=0.1, T=100, sigma=1,
                       start_act=None, seed=None):
        if start_act is None:
            #start_act=np.zeros(self.Nneuron)
            start_act=self.steady_state(input_h_mean)

        num_steps = int(T/dt)
        sqrtdt=np.sqrt(dt)
        rng = np.random.default_rng(seed=seed)

        res = []

        res.append(start_act)
        #RK like [AJ Roberts, aXriv, 2012]
        # for istep in range(num_steps):
        #     act_t= np.copy(res[-1])

        #     S = np.random.rand(self.Nneuron)
        #     S[S>=0.5]=1
        #     S[S<0.5]=-1
        #     dW=sqrtdt*rng.normal(size=self.Nneuron)
        #     K1 =  dt*(-self.tau_r*act_t + input_h_mean +
        #             self.W_connect @ act_t )  + sigma*(dW-sqrtdt*S)
        #     K12 = act_t + K1
        #     K2 = dt*(-self.tau_r*K12 + input_h_mean +
        #             self.W_connect @ K12   )  + sigma*(dW+sqrtdt*S)
        #     act_new = act_t + 0.5*(K1+K2)
        #     res.append(act_new)

        #Euler-Maruyame
        for istep in range(num_steps):
            act_t= np.copy(res[-1])


            dW=sqrtdt*rng.normal(size=self.Nneuron)
            K1 =  dt*(-self.tau_r*act_t + input_h_mean +
                    self.W_connect @ act_t )  + sigma*(dW)

            act_new = act_t + K1
            res.append(act_new)

        return np.asarray(res)


    def time_dep_input_dW_color(self, input_h_mean,
                       dt=0.1, T=100, sigma=1,
                       start_act=None, seed=None,
                       cov_noise = None):
        if start_act is None:
            #start_act=np.zeros(self.Nneuron)
            start_act=self.steady_state(input_h_mean)
        if cov_noise is None:
            cov_noise=np.eye(self.Nneuron)

        num_steps = int(T/dt)
        sqrtdt=np.sqrt(dt)
        rng = np.random.default_rng(seed=seed)

        res = []

        res.append(start_act)
        #Euler-Maruyama
        for istep in range(num_steps):
            act_t= np.copy(res[-1])


            dW=sqrtdt*rng.multivariate_normal(mean=np.zeros(self.Nneuron),
                                              cov=cov_noise)
            K1 =  dt*(-self.tau_r*act_t + input_h_mean +
                    self.W_connect @ act_t )  + sigma*(dW)

            act_new = act_t + K1
            res.append(act_new)

        return np.asarray(res)

    def transform_sigma_input(self, sigma_in):
        return self.Interaction_mat @ sigma_in @ self.Interaction_mat.T


class W_recurrent:
    def __init__(self,
            N_neurons,
            mean_W=0.5,
            std_W=1,
            mode="random_normal",
            norm=0.01,):
        self.N_neurons=N_neurons
        self._init_connectivity(mean_W, std_W,mode)
        self.normalize_connectivity(norm)

    def _init_connectivity(self, mean, std, mode):
        if "random_normal" in mode:
            self.connectivity = (np.random.randn(self.N_neurons, self.N_neurons)*std)+mean

    def normalize_connectivity(self, norm ):
        current_norm = self.calc_norm()
        self.connectivity *= norm/current_norm

    def calc_norm(self, W=None):
        if W is None:
            W=self.connectivity
        return np.sqrt(np.sum(np.square(W)))

    def calc_EigvalMax(self, W=None):
        if W is None:
            W=self.connectivity
        return np.max(linalg.eigvals(W).imag)

    #plot W
    def plot_connectivity(self, W=None, returnim=False):
        if W is None:
            W=self.connectivity
        vmin=np.min(W)
        fig=plt.figure()
        if vmin<0:
            plt.imshow(W,
                 norm=colors.TwoSlopeNorm(0, vmin=vmin, vmax=np.max(W)),
                 cmap="RdBu_r", interpolation='none')
        else:
            plt.imshow(W,vmin=0,
                 cmap="Reds", interpolation='none')
        plt.colorbar()
        if returnim: return fig

def create_W(n_neuron,
             mean_connect,
             spectral_bound=0.99,
             symmetric=True,
             var_conn=1,
             seed=1):

    rng = np.random.default_rng(seed=seed)
    w_connect=rng.normal(mean_connect, scale=var_conn/np.sqrt(n_neuron),
                      size=(n_neuron, n_neuron))
    if symmetric:
        w_connect= w_connect + w_connect.T
        w_connect -= np.mean(w_connect)
        w_connect /= np.std(w_connect)
        w_connect *= var_conn/np.sqrt(n_neuron)
        w_connect += mean_connect

    eig=linalg.eigvals(w_connect)
    max_eig=np.max(eig.real)
    w_connect = w_connect/max_eig*spectral_bound
    return w_connect

def create_W_v2(n_neuron,
             mean_connect=0,
             spectral_bound=0.99,
             symmetric=True,
             exc_proportion=0.8,
             seed=1):
    w_inhib=1
    w_exc = mean_connect/(n_neuron*exc_proportion) + w_inhib*(
        1-exc_proportion)/exc_proportion
    print(w_inhib, w_exc)

    rng = np.random.default_rng(seed=seed)
    w_ident=rng.random(size=(n_neuron, n_neuron))
    w_ident[np.triu_indices(n_neuron, k=1)]= w_ident.T[np.triu_indices(n_neuron, k=1)]

    w_connect = np.empty_like(w_ident)*np.nan
    w_connect[w_ident<exc_proportion]=w_exc
    w_connect[w_ident>=exc_proportion]=-1*w_inhib
    print(w_connect.mean(), np.sum(w_connect<0)/n_neuron**2)

    eig=linalg.eigvals(w_connect)
    max_eig=np.max(eig.real)
    w_connect = w_connect/max_eig*spectral_bound
    return w_connect

#%%
if __name__ == '__main__':
    #get W
    Nneuron=100
    W_inst = W_recurrent(Nneuron,
                         mean_W=0.0, std_W=10,
                         norm=0.8)
    W_rec = W_inst.connectivity

    #create network instance
    nn_rec = LinearRecurrentNetwork(
        W_rec,
        )

    #create input
    input_alpha = np.random.rand(Nneuron)
    steady_state=nn_rec.steady_state(input_alpha)


    plt.scatter(input_alpha, steady_state)
    plt.xlabel("Input")
    plt.ylabel("Steady state")
    plt.show()

    plt.imshow(W_rec, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.show()



