#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ozgesahin
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse import diags
from a11e02pyramid import a11e02matrixelement,a11e02pyramid
from matplotlib import cm
# given f function 
def f(x,y) : 
    return (0<=x) * (0<=y) * (x<=1./2)*(y<=1./2)

def surface_plot(N,Uh,h):
    
    # Generate mesh for plotting
    grid_1D = np.linspace(0+h,1-h,N)
    X,Y = np.meshgrid(grid_1D,grid_1D)
    
    # Create surface plot for Poisson equation
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Uh,cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Fem solution for 2D Poisson equation")
    
    # Show plot
    plt.show()
   
def a12e02Poisson2D(N) :
   h = 1.0/(N+1) # 
   i = np.arange(1,N+1) 
   j = np.arange(1,N+1)  
   F = np.zeros((N*N)) # load vector
   Uh = np.zeros((N,N)) # approx. solution of u
   I = sparse.eye(N)
   B = diags([-1,4,-1], [-1, 0, 1], shape=(N,N)) 
   S = diags([-1, -1],[-1 ,1],shape=(N,N))
   K = sparse.kron(I,B)+sparse.kron(S,I).toarray() # stiffness-matrix 
   c = a11e02matrixelement(1,1,1,1,h,0) 
   # mass-matrix (for simplicity, overlaps over the elements for instance 1-2 or 2-3  .. neglected since their values are nearly zero  )
   M = diags([c],[0],shape=(N*N,N*N)).toarray()  
   index = 0
   total =np.zeros((N*N,N*N))
   grid_1D = np.linspace(0+h,1-h,N) # values of nodes (x,y)
   X,Y = np.meshgrid(grid_1D,grid_1D)
   for jj in j : 
        for ii in i : 
            # total matrix consists of all values of shape function of node on space 
            total[:,index] = a11e02pyramid(ii,jj,h,0)(X,Y).reshape((1,N*N))
            # load vector 
            F[index] = f(ii*h,jj*h)  
            index = index +1 
  
   #solving the eq. KU = MF 
   U = spsolve(K,np.dot(M,F)) # U is the coeff.
    
   index = 0
   for jj in j:
      for ii in i:
         
         Uh[ii-1,jj-1] = np.dot(U,total[index,:]) # finding app. solution U = sum(Ui*wi)
         index = index +1
   surface_plot(N,Uh,h) # plot 

a12e02Poisson2D(3)   

