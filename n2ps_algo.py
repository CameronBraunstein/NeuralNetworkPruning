# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:04:02 2018

@author: Abhishek Nair
"""

import numpy as np
import math

def sigmoid(x):
    return float(1)/(1+math.e**(-x))

def sig_calc(X,W):
    num_inp_neurons = X.shape[1]
    txip = X.sum(axis=0)
    ftxip = sigmoid(txip)
    s_inp = np.zeros(num_inp_neurons)
    alpha = 0
    for i in range(num_inp_neurons):
        for j in range(W.shape[1]):
            s_inp[i] += abs(ftxip[i] + W[i][j])
        alpha += s_inp[i]    
    alpha = alpha/num_inp_neurons
    
    return s_inp,alpha,ftxip

def generate_inp_neurons(X,W):
    si,alpha,ftxip = sig_calc(X,W)
    for i in range(X.shape[1]):
        if si[i] <= alpha:
            for j in range(W.shape[1]):
                W[i][j] = 0
    return W

def generate_hid_neurons_layer(X_inp,inp_Wt,out_Wt,layer):
    num_hid_neurons = inp_Wt.shape[1]
    tnetjl = np.zeros(num_hid_neurons)
    
    if layer == 1: 
        for i in range(X_inp.shape[0]):
            for j in range(X_inp.shape[1]):
                tnetjl += X_inp[i][j]* inp_Wt[j]
    else:
        for i in range(X_inp.shape[0]):
            tnetjl += X_inp[i]* inp_Wt[i]       
      
    si = np.zeros(num_hid_neurons)
    ftnet = np.zeros(num_hid_neurons)
    
    for s in range(num_hid_neurons):
        ftnetjl = sigmoid(tnetjl[s])
        for m in range(out_Wt.shape[1]):
            si[s] += abs(ftnetjl + out_Wt[s][m])
        ftnet[s] = ftnetjl
    
    beta = si.sum(axis=0)
    beta = beta/num_hid_neurons    
    for i in range(num_hid_neurons):
        if si[i] <= beta:
            for k in range(inp_Wt.shape[0]):
                inp_Wt[k][i] = 0
            for j in range(out_Wt.shape[1]):
                out_Wt[i][j] = 0

    return out_Wt,ftnet

X =np.random.rand(3,2)
W1 = np.random.rand(2,3)
W2 = np.random.rand(3,3)
W3 = np.random.rand(3,1)

inp_Wt1 = generate_inp_neurons(X,W1)

hid_Wt1,ftnet = generate_hid_neurons_layer(X,inp_Wt1,W2,1)

hid_Wt2,ftnet_h = generate_hid_neurons_layer(ftnet,hid_Wt1,W3,2)
