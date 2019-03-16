#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


4a)



  -1/h**2 (IN ⊗ S + S ⊗ IN )uh =-1/h**2((IN ⊗ S)*uh + (S ⊗ IN)*uh )
   = -1/h**2((IN*UH*S`)+(S*UH*I`)
   = -1/h**2((IN*UH*S)+(S*UH*I)
   IN*UH = UH = UH*I
   -1/h**2((UH*S)+(S*UH)) = FN


"""

#4b)

from scipy.sparse import diags
import numpy as np
from scipy import sparse
from scipy.linalg import solve_sylvester

def a06e04silsolver(p,i):
    N = 2**p - 1
    h = 1./(N+1)     
    xh_ = np.linspace(0,1,N+2)    
    xh = xh_[1:-1]                
    yh_ = np.linspace(0,1,N+2)
    yh = yh_[1:-1]
    if i == 1:
        f = lambda x, y: 12*x*y - 6*x*y**3 - 6*y*x**3
    elif i == 2:
        f = lambda x, y: 10*np.pi**2*np.sin(3*np.pi*x)*np.sin(np.pi*y)
    
    fh = np.zeros(N**2)
    s = 0
    for k in range(0, len(yh)):
        for j in range(0, len(xh)):
            fh[s] = f(xh[j], yh[k]) 
            s += 1
    Fh = np.resize(fh,(N,N)) 
    if N == 1 :
       S = sparse.csr_matrix(np.array((-2))).todense()
    else : 
       S =diags([1, -2, 1], [-1, 0, 1], shape=(N,N)).todense() 
    Sh = (-1/h**2)*S
    Uh = solve_sylvester(Sh,Sh,Fh)
    uh = np.resize(Uh,(1,N**2))
    return uh

