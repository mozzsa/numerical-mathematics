#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from a04e04getPDE import a04e04getPDE
from a06e04silsolver import a06e04silsolver
from scipy.sparse import linalg
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import pandas 
import time 

def a04e04solve(i):
   stepsize =[] 
   errors = [] 
   errorssil = []
   EOCp = []
   EOCpsil = []
   tt = []
   ttsil = []
   
   for k in range(1,10,1):
      N = 2**k - 1
      h = 1./(N+1)   #step size
      xh_ = np.linspace(0,1,N+2)    #N+2 because we want to have N points in BETWEEN (0,1)
      xh = xh_[1:-1]                #only taking the inner points!
      yh_ = np.linspace(0,1,N+2)
      yh = yh_[1:-1]
#Setting up exact solution 1 and 2
      if i == 1:
         u = lambda x, y: x*y + x**3*y**3 - x**3*y - x*y**3
      elif i == 2:
         u = lambda x, y: np.sin(3*np.pi*x)*np.sin(np.pi*y) 
      
      u_exact = np.zeros(N**2)
      s = 0
      for t in range(0, len(yh)):
         for j in range(0, len(xh)):
            u_exact[s] = u(xh[j], yh[t]) #Calculating exact solution for all pairs (xh,yh)
            s += 1      
        
      time1 = time.clock()
      [Lh, fh] = a04e04getPDE(k,i)   #Solving u_h for every (k,i)
      u_h = linalg.spsolve(Lh, fh)
      time2 = time.clock()
      tt.append(time2-time1)
      u_hsil = a06e04silsolver(k,i) 
      time3 = time.clock()
      ttsil.append(time3-time2)
      
        
      diff = u_h - u_exact
      diffsil = u_hsil - u_exact
      err = np.max(np.abs(diff))
      errsil = np.max(np.abs(diffsil))
      if k > 1 :  
         eoc = (np.log10(err) - np.log10(errors[-1]))/(np.log10(h) - log10(stepsize[-1]))
         eocsil =(np.log10(errsil) - np.log10(errorssil[-1]))/(np.log10(h) - log10(stepsize[-1]))   
         EOCp.append(eoc)
         EOCpsil.append(eocsil)
      errors.append(err)
      errorssil.append(err)
      stepsize.append(h)
   str = ""
   if i == 1:
      str = " Function1"
      
   elif i == 2:
      str = " Function2"
     
   plt.figure()
   plt.loglog(stepsize,errors)
   plt.title('Error plot over step size h'+str)
   plt.xlabel('h')
   plt.ylabel('error')
   plt.show()
   plt.figure()
   plt.loglog(stepsize,errorssil)
   plt.title("Error(relies on solving the Sylvester equation)"+str)
   plt.xlabel('h')
   plt.ylabel('error')
   plt.show()
   print("Experimental order of convergence"+str)
   print(pandas.DataFrame(EOCp,range(2,10,1)))
   print("EOCp(relies on solving the Sylvester equation)"+str)
   print(pandas.DataFrame(EOCpsil,range(2,10,1)))
   plt.figure()
   plt.plot(stepsize,tt,'g--', stepsize, ttsil,'b:')
   plt.title("Runtime over h,Sylvester equation(blue)"+str)
   plt.show()
   return errors,errorssil

a04e04solve(1)
#For function 1 while h is increasing oppositely error is decreasing.
#There is no explicit difference between the errors relies on solving the Sylvester eq. or from the normal way
#EOCs are different when h is increasing  error is decreasing much faster with Sylvester method 
#Runtime is definitely better if we use Slyvester method when h is too small
a04e04solve(2)
#For function 2 while h is increasing , error is increasing as well.
#There is no explicit difference between the errors relies on solving the Sylvester eq. or from the normal way
#EOCs are nearly similar 
#Runtime is definitely better if we use Slyvester method when h is too small

