#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


u_exact = x**4-2*x**2+1./2
beta = 0



"""
from a06e03getPDE import a06e03getPDE
import matplotlib.pyplot as plt
beta = 0
errors = []
h_values = []
for p in range(2,10):
   N = 2**p - 1
   h = 1./(N+1)
   xh_ = np.linspace(-1,1,N+2)    
   xh = xh_[1:-1]  
   Lh,fh = a06e03getPDE(p,beta)
   uh = linalg.spsolve(Lh, fh)
   u = lambda x : x**4-2*x**2+1./2
   u_exact = u(xh)
   diff = uh-u_exact
   err = np.max(np.absolute(diff))
   #err = np.linalg.norm(diff, np.inf)
   errors.append(err)
   h_values.append(h)
   
plt.figure()
plt.loglog(h_values,errors)
plt.title('Error plot over step size h')
plt.xlabel('h')
plt.ylabel('error')
plt.show()


#3c)
# numerical solution behaviour changes with different choices of beta
p=5
N = 2**p - 1
h = 1./(N+1)
xh_ = np.linspace(-1,1,N+2)    
xh = xh_[1:-1] 
for i in (0,10,20,-10,-20):   
   Lh,fh = a06e03getPDE(5,i)    
   uh = linalg.spsolve(Lh, fh)
   plt.figure()
   plt.plot(xh,uh)
   plt.title("Numerical Solution for different beta: %d p = 5 "%i)
   plt.show()
    