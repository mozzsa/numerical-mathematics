#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ozgesahin
"""

import numpy as np
import matplotlib.pyplot as plt
from a09e02getpoly import a09e02getpoly
#exact solution of u
def u(x):
   return np.exp(x)+(-1-np.exp(1))*x-1

error = [] #error values
list_N=np.arange(2,11) #N = (2,3,....,10)
for N in list_N :
   print(N)
   xh_= np.linspace(0,1,N+1)
   xh=xh_[1:] #exclude 0
   uh= u(xh) 
   eps = a09e02getpoly(N)
   err = uh-eps
   error.append(np.linalg.norm(err, np.inf))

#loglog-plot for some different N values    
plt.loglog(list_N,error)
plt.title('Error plot over step size N')
plt.xlabel('N')
plt.ylabel('Error')
plt.show()

