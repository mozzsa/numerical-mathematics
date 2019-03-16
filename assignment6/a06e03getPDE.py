#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import diags,linalg


def a06e03getPDE(p,beta):
    N = 2**p - 1
    h = 1./(N+1)     
    xh_ = np.linspace(-1,1,N+2)    
    xh = xh_[1:-1] 
    fh = np.zeros(N)
    fh[0] = fh[0]-(1./2*h**2)*(4-2*xh[0]**2+xh[0]*h)*(-1./2)
    fh[N-1] = fh[N-1] - (1./2*h**2)*(4-2*xh[0]**2-xh[0]*h)*beta*h
    center_func = lambda x,h: -8+4*x**2+32*h**2
    upper_func =  lambda x,h: 4-2*x**2-x*h
    lower_func =  lambda x,h: 4-2*x**2+x*h
    center = center_func(xh,h)
    upper =  upper_func(xh[0:-1],h)
    lower =  lower_func(xh[1:len(xh)],h)
    center[-1] += upper_func(xh[-1],h)
    Lh = diags([lower,center,upper], [-1, 0, 1], shape=(N,N))
    return Lh,fh