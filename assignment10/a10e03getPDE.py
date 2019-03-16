#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: ozgesahin

2a)
if we take w = (beta - alpha ) / (b-a) (x-a) which satisfies the boundary condition
let say u_d = u - w 
then we get transformed eq. with homogeneous boundary condition
-u_d" + u_d = f-w
u_d(a) = 0
u_d(b) = 0

variational formulation is :
integral(u`_d*v'+u_d*v,a,b)=integral(f-w)*v,a,b)  v(a) = 0 v(b)=0

"""

import numpy as np
from scipy.sparse import diags
from scipy import integrate
from scipy.sparse.linalg import spsolve

def a10e03getPDE(a,b,alpha,beta,f,N):
   #the grid (a, a + h, . . . , b) of N + 2 uniformly distributed points of [a, b]
   xh = np.linspace(a,b,N+2)
   h = xh[1]-xh[0] # uniform grid 
   # finite element approximation
   center = [] # main diagonal
   left = [] # left or right diagonal(same) 
   fh = [] # right hand side of the linear system
   for i in np.arange(1,len(xh)-1):
       # right hand side of the linear system using the trapezoidal rule
       v_l = lambda x:(x-xh[i-1]/h) #left side of base function 
       v_r = lambda x:(xh[i+1]-x/h) # right side of base function  ### else 0
       v_lsquare = lambda x:(x-xh[i-1]/h)**2  
       v_rsquare = lambda x:(xh[i+1]-x/h)**2
       v_csquare = lambda x:((x-xh[i]/h)*(xh[i+1]-x/h))**2
       w = lambda x : (beta-alpha)/(b-a) *(x-a)
       temp_f = h*((v_l(xh)+v_r(xh))*(f(xh)-w(xh)))#trapezoidal rule
       temp_f[0]= temp_f[0]/2.
       temp_f[-1]= temp_f[-1]/2. 
       fh.append(sum(temp_f))
       #main diagonal (i=j)
       ans_l,err = integrate.quad(v_lsquare,xh[i-1],xh[i])
       ans_r,err = integrate.quad(v_rsquare,xh[i],xh[i+1])
       center.append(2./h + (ans_l + ans_r))
       #left or right diagonal (i-j = 1)
       ans,err = integrate.quad(v_csquare,xh[i],xh[i+1])
       left.append(ans-(1./h))
   diagonals=[]
   diagonals.append(left)  
   diagonals.append(center)
   diagonals.append(left)
   if N==1: #1 by 1 matrix
      K = np.array(center)
   else : 
      K = diags(diagonals, [-1, 0, 1], shape=(N,N)) # sparse  matrix
   U_ = spsolve(K, fh) # solving the linear equation
   U = np.zeros((len(U_)+2)) # adding boundary values to vector U
   U[1:len(U)-1]  = U_
   U[0] = alpha
   U[-1]  = beta # Now U is numerical solution of transformed func. u_d
   # u = u_d + w 
   U = U + w(xh) # U is the solution of original eq with non homogeneous boundary 
   return xh,U 