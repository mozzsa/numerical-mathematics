#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ozgesahin


2a)

exact solution to (1) -> u = e**x+(-1-e)*x-1

2b)

variational formulation of (1) ->

∫u'v' = ∫-e**xv for all v∈C'([0,1]) and v(0)=0
 
2c) 

explained in the provided code

2f)

All the elements of diagonals have to calculated.Because of that it is not favorable.

"""
#returns coefficients ξ1, . . . , ξN
import numpy as np
from scipy.sparse import diags
from scipy import integrate
from scipy.sparse.linalg import spsolve

def a09e02getpoly(N):
    xh = np.linspace(0,1,N+1) 
    center = [] # main diagonal
    left = [] # or right diagonal they are same 
    fh = [] # f array
    for i in np.arange(1,len(xh)-1):
       ans,err = integrate.quad(lambda x:(i*x**(i-1))**2,xh[i-1],xh[i+1]) # i=j
       center.append(ans)
       if i != len(xh)-2 :
          ans,err = integrate.quad(lambda x:i*x**(i-1)*(i+1)*x**i,xh[i],xh[i+1]) # abs(i-j)=1
          left.append (ans)
       #elsewhere the elements 0
       ans,err =integrate.quad(lambda x:x**i*np.exp(x)*-1,0,1) 
       fh.append(ans) # right side of the eq.                           
    diagonals=[]
    diagonals.append(left)  
    diagonals.append(center)
    diagonals.append(left)
    if N>1: 
       if N==2: #1 by 1 matrix
          K = np.array(center)
       if N>2 : 
          K = diags(diagonals, [-1, 0, 1], shape=(N-1,N-1))
       epsilon = spsolve(K, fh)
       coeff = np.zeros((len(epsilon)+1))
       coeff[0:len(coeff)-1]  = epsilon
       coeff[-1]  = -1.0/N  #due to the boundary condition epsilon ξN = -1/N  
    else:
       coeff = -1.0/N                             
    return coeff




