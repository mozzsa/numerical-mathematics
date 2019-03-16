
from scipy.sparse import linalg
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import pandas 
from a04e04getPDE import a04e04getPDE

def a04e04solve(i):
   stepsize =[] 
   errors = [] 
   EOCp = []

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
      
      [Lh, fh] = a04e04getPDE(k,i)   #Solving u_h for every (k,i)
      u_h = linalg.spsolve(Lh, fh)
      u_exact = np.zeros(N**2)
      s = 0
      for t in range(0, len(yh)):
         for j in range(0, len(xh)):
            u_exact[s] = u(xh[j], yh[t]) #Calculating exact solution for all pairs (xh,yh)
            s += 1
        
      diff = u_h - u_exact
      err = np.linalg.norm(diff, np.inf)#Calculating maximum error norm
      if k > 1 :  
         eoc = (np.log10(err) - np.log10(errors[-1]))/(np.log10(h) - log10(stepsize[-1]))
         EOCp.append(eoc)
      errors.append(err)
      stepsize.append(h)
   if i == 1:
      print("Function1")
   elif i == 2:
      print("Function2")
   plt.loglog(stepsize,errors)
   plt.title('Error plot over step size h')
   plt.xlabel('h')
   plt.ylabel('error')
   plt.show()
   print("Experimental order of convergence")
   print(pandas.DataFrame(EOCp,range(2,10,1)))
   return errors



