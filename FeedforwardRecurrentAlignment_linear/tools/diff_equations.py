#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:36:16 2022

@author: sigridtragenap
"""

import numpy as np

def fprime_Vmembr(t,v, W_rec, input_h, nonlinearity, alpha=1):
	#term_decay=-alpha*v
	#term_recurrence=W_rec @ nonlinearity(v)
	#print(term_decay.shape, term_recurrence.shape[0])
	return (-alpha*v + input_h + W_rec @ nonlinearity(v))

def fprime_Vmembr_mat(t,V_vector, W_rec, input_M, nonlinearity, alpha=1):
	assert len(input_M.shape)==2
	Nneuron = W_rec.shape[0]
	M=input_M.shape[1]
	V_mat = V_vector.reshape(Nneuron, M)
	update = (-alpha*V_mat + input_M + W_rec @ nonlinearity(V_mat))
	return update.flatten()


def fprime_rate(t,v, W_rec, input_h, nonlinearity, alpha=1):
    #nonlinearity is around input and W*v
    #instead of just around v
	return (-alpha*v + nonlinearity(input_h + W_rec @ v))

def fprime_rate_mat(t,V_vector, W_rec, input_M, nonlinearity, alpha=1):
	assert len(input_M.shape)==2
	Nneuron = W_rec.shape[0]
	M=input_M.shape[1]
	V_mat = V_vector.reshape(Nneuron, M)
	update = (-alpha*V_mat + nonlinearity(input_M + W_rec @ V_mat))
	return update.flatten()

def fprime_W_sigma3(t,W_rec, epsilon, mu, V_mat, nonlinearity):
	assert len(V_mat.shape)==2
	Nneuron = V_mat.shape[0]
	M_inv=1./V_mat.shape[1]
	W_mat = W_rec.reshape(Nneuron, Nneuron)
	update= epsilon*(M_inv * nonlinearity(V_mat) @ nonlinearity(V_mat.T) - mu*W_mat)
	return update.flatten()


