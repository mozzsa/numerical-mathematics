#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

a)

U_l^j = f(x_j) + k*sum[m=1 - l](V_j^m)[2]  (2 means the second entry of the vector)

b)
"""

import numpy as np

def a08e03LaxWend(t,xmin,xmax,c,f,g,h,k) :
    #h,k step size
    # number of time steps
    l = int(t/k)
    xi_ = np.linspace(xmin, xmax, (xmax-xmin)/h+1) #grid of spatial coordinate
    j = len(xi_)
    xmin = xmin - l*h
    xmax = xmax + l*h
    xi = np.linspace(xmin, xmax, (xmax-xmin)/h+1)
    M = np.array(([0,-c],[-c,0])) # M matrix
    lamb = k/h # fix ratio for LaxWend
    # finds the initial value of V
    v1init = c*(f(xi)-f(xi-h))/h #Vector for the c*f'
    v2init = g(xi)
    Vinit = np.zeros((2,len(xi))) #1st entry: c*f'(x_j); 2nd entry: g(x_j)
    Vinit[0,:] = v1init
    Vinit[1,:] = v2init
    V = np.zeros((l,j))
    # dimension
    d = M.shape[1]
    #LaxWend method 
    coef1 = np.dot(M,M)*lamb**2
    coef2 = M*lamb
    #calculating V Vector for l=1 until l=l
    for ind in range(1,l+1) :
        VV = np.dot(np.eye(d) - coef1, Vinit[:,1:-1]) + 0.5*np.dot(coef1 - coef2, 
                Vinit[:,2:]) + 0.5*np.dot( coef1 + coef2, Vinit[:,:-2])
        Vinit = VV
        V[ind-1,:] = Vinit[1,l-ind:l-ind+j]
    #calculating U_l^j (for all x_j E [xmin, xmax] at t=t_l)
    U = f(xi_)+k*np.sum(V,axis=0)
    return U 
    

   