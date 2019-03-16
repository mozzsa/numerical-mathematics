#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scipy.sparse import diags
import numpy as np
from scipy import sparse
 

def a04e04getPDE(p,i):
    N = 2**p - 1
    h = 1./(N+1)         #step size
    xh_ = np.linspace(0,1,N+2)    #N+2 because we want to have N points in BETWEEN (0,1)
    xh = xh_[1:-1]                #only taking the inner points!
    yh_ = np.linspace(0,1,N+2)
    yh = yh_[1:-1]
    #Setting up for function 1 and 2
    if i == 1:
        f = lambda x, y: 12*x*y - 6*x*y**3 - 6*y*x**3
    elif i == 2:
        f = lambda x, y: 10*np.pi**2*np.sin(3*np.pi*x)*np.sin(np.pi*y)
    
    fh = np.zeros(N**2)
    s = 0
    for k in range(0, len(yh)):
        for j in range(0, len(xh)):
            fh[s] = f(xh[j], yh[k]) #Calculating all f-values for all pairs (xh,yh) and setting up f-vector
            s += 1
    NN = N**2
    if N == 1 :
       L_h = sparse.csr_matrix(np.array((-4)))
    else :
       main_diag = np.ones(NN)*-4.0
       side_diag = np.ones(NN-1)
       side_diag[np.arange(1,NN)%N==0] = 0
       up_down_diag = np.ones(NN-N)
       diagonals = [main_diag,side_diag,side_diag,up_down_diag,up_down_diag]
       L_h = diags(diagonals, [0,-1,1,N,-N], format="csr")
    Lh = (-1./h**2)*L_h
    return Lh, fh

