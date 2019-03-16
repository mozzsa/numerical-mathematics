#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ozgesahin
"""

from scipy.sparse import diags
import numpy as np 
from scipy import sparse
from matplotlib import animation
import matplotlib.pyplot as plt
import time 


#when t = 0 initial condition
def func_t(x):
    return np.exp(-9*(x-1/2)**2)

#animation
def init(line,):
    line.set_data([], [])
    return (line,)

def animate(i,x,U,ax,line):
    y = U[i,:]
    ax.set_ylim((min(y),max(y)))
    line.set_data(x, y)
    return (line,)

plt.figure()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
line, = init(line,)


def a12e03Heat(M,N,mov):
    
    # N time steps
    t_ = np.linspace(0,1,N+2)
    delta_t = t_[1]-t_[0]
    t = t_[1:-1]
    
    # M interior nodes
    x_ = np.linspace(0,1,M+2)
    h= x_[1]-x_[0]
    x  = x_[1:-1] 
    
    # M =~ h*I  # mass matrix, galerkin finite element 
    A = 1.0/h * diags([-1,2,-1],[-1,0,1],shape=(M,M)).toarray() # stiffness matrix
    
    init = func_t(x) # when t = 0 Uj-1
    
    U = np.zeros((N,M))
    I = np.zeros(M)
    
    for i in np.arange(len(t)) : # Backward Euler method
        U_ = init+(delta_t/h)*(1./100*I-np.dot(A,init)) # U_ is the approx. solution at th Uj+1
        U[i,:] = U_
        init = U_ 
        
    if mov == 1 : # animation
        anim = animation.FuncAnimation(fig, animate,fargs=(x,U,ax,line,),
                               frames=len(t), interval=11, blit=True)
        anim.save('animation_heat.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 
        
    return x,U
    
#a12e03Heat(5,6,1) 
#a12e03Heat(5,6,0)

#computational time 
time_list = []
k_list = [3,4,5,6]

for k in k_list:
    M = 2**k
    N = 2**k
    t0 = time.time()
    X,U = a12e03Heat(M,N,0) 
    t1 = time.time()
    time_list.append(t1-t0)
    
    
#plot
plt.figure()
plt.plot(k_list,time_list)
plt.title("Computational time")
plt.show()
    
    

