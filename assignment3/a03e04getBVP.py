

from scipy.sparse import diags
import numpy as np

def a03e04getBVP(p):
    N = 2**p - 1
    h = 1./(N+1)         #step size
    
    xh_ = np.linspace(0,1,N+2)    #N+2 because we want to have N points in BETWEEN (0,1)
    xh = xh_[1:-1]       #interior points of the grid
    
    f = lambda x: -x**3 + 15*x**2 - 18*x - 5             #f(x)
    fh = np.zeros(N)
    fh = f(xh)                                          #calculate fh for all grid points in between x element (0,1)
#    fh[0] = fh[0] + 1./h**2 - 2./h
#    fh[-1] = fh[-1] + 3./h**2 + 6./h
    
    Lh = 1./(2*h**2) * diags([-2+4*h, 4+2*h**2, -2-4*h], [-1, 0, 1], shape=(N,N))
    
    return xh,Lh,fh