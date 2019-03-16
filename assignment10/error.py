#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: ozgesahin
"""
from a10e03getPDE import a10e03getPDE
from a10e03getPDEBonus import a10e03getPDEBonus 
import numpy as np
import matplotlib.pyplot as plt

#2c)

def f(x):
    return np.power(x, 3)-6.*x+(np.pi**2/4.+1)*np.cos(np.pi/2.*x)
 
#solution of u_d is np.cos(np.pi/2*x)+np.power(x, 3) with the boundary condition we take w = x so 
# u = u-d+w = np.cos(np.pi/2*x)+np.power(x, 3)+x
def u_exact(x):
    return np.cos(np.pi/2*x)+np.power(x, 3)+x
    
a = -1
alpha = -1
beta = 1
b = 1 
errors_num = [] # errors
EOCp_num = []
stepsize = []
for p in np.arange(1,8):
   N = 2**p - 1
   x,U = a10e03getPDE(a,b,alpha,beta,f,N) # finite element solution
   h = x[1]-x[0]
   uh_exact = u_exact(x) # exact solution
   diff = uh_exact - U # difference between u_exact and numerical solution
   err = np.linalg.norm(diff, np.inf) #Calculating maximum error norm 
   #EOC
   if p > 1 :  
      eoc = (np.log10(err) - np.log10(errors_num[-1]))/(np.log10(h) - np.log10(stepsize[-1]))
      EOCp_num.append(eoc)
   errors_num.append(err)
   stepsize.append(h)
        
print("Experimental order of convergence")
print(EOCp_num)
plt.figure()
plt.loglog(stepsize,errors_num)
plt.title('Error plot over step size h')
plt.xlabel('h')
plt.ylabel('error')
plt.show()

#the error is decreasing as h is increasing and absolute value of rate of convergence is decreasing as h decreasing .

#2e)
    
def c(x):
    return np.log(x+4)

def f_(x):
    return np.exp(x/4.)

a = -np.pi
alpha = -1
beta = 1
b = np.pi 
errors_num = [] # errors
EOCp_num = []
stepsize = []
for p in np.arange(1,8):
   N = 2**p - 1
   x,U = a10e03getPDEBonus(a,b,c,alpha,beta,f_,N) # finite element solution
   h = x[1]-x[0]
   uh_exact = u_exact(x) # exact solution  ???? u_exact function is missing ..
   diff = uh_exact - U # difference between u_exact and numerical solution
   err = np.linalg.norm(diff, np.inf) #Calculating maximum error norm 
   #EOC
   if p > 1 :  
      eoc = (np.log10(err) - np.log10(errors_num[-1]))/(np.log10(h) - np.log10(stepsize[-1]))
      EOCp_num.append(eoc)
   errors_num.append(err)
   stepsize.append(h)
        
print("Experimental order of convergence")
print(EOCp_num)
plt.figure()
plt.loglog(stepsize,errors_num)
plt.title('Error plot over step size h')
plt.xlabel('h')
plt.ylabel('error')
plt.show()
