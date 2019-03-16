

from a03e04getBVP import a03e04getBVP
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

def a03e04solve():
    
    u = lambda x: 1 + 3*x**2 - x**3     #exact solution
    err_max = np.zeros(14)
    h = np.zeros(14)
    
    for i in range(2, 16, 1):
        [xh,Lh,fh] = a03e04getBVP(i)
        h[i-2] = 1./2**i
        
        fh[0] = fh[0] + 1./h[i-2]**2 - 2/h[i-2]
        fh[-1] = fh[-1] + 3./h[i-2]**2 + 6./h[i-2]
        
        u_h = sparse.linalg.spsolve(Lh, fh)   #compute FDM solution
        
        err = []
        err = u_h - u(xh)
        err_max[i-2] = np.linalg.norm(err,np.inf)
        
        
        
    plt.loglog(h,err_max)
    plt.title('Error plot over step size h')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.show()


#  the error is going to zero for h -> 0