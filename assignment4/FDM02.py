"""
A not so well-behaved FDM example.

How to use: 
On the command line type
python FDM02.py
"""

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import sys

# parameter values
a_list = [0.1, 0.05, 0.01]
N_list = [10, 20, 40, 80] 

# exact solution
def u(x,a):
    return (1 - np.exp(-x/a)) / (1 - np.exp(-1./a))

# generates matrix Lh in dependence of a and N
def L_h(a,N):
    h = 1./(N + 1)
    Lh = (-a/h**2*sp.diags([1, -2, 1], [-1, 0, 1], shape=(N,N), format="csr") -
           1./2/h*sp.diags([-1, 0, 1], [-1, 0, 1], shape=(N,N), format="csr"))
    return Lh

# generates r.h.s fh in dependence of a and N
def f_h(a,N):
    h = 1./(N + 1)
    fh = np.zeros(N)            # no inhomogenity
    fh[-1] = a/h**2 + 1./2./h   # boundary condition at 1
    return fh

# computes FDM solution for given a and N
def u_h(a,N):
    A = L_h(a,N) # get FDM matrix
    b = f_h(a,N) # get r.h.s
    return spsolve(A,b)    

# generate table with norms of L_h_inv
print '\nMatrix norms\n'

print ' N \\ a',
for a in a_list:
    print ' %.2f ' % a,
print '\n',
print '-------'*(len(a_list) + 1)
for N in N_list:
    print ' %4d ' % N,
    for a in a_list:
        Lh = L_h(a,N) 
        print ' %.2f ' % np.linalg.norm(np.linalg.inv(Lh.toarray()),np.inf),
    print '\n',


# plot exact solution for different parameter values a
grid2 = np.linspace(0,1,201)
plt.figure(1)
for a in a_list:
    plt.plot( grid2, u(grid2,a),label='a = %.2f' % a)
plt.ylim([0,1.4])
plt.xlabel('x')
plt.ylabel('u(x,a)')
plt.title('exact solution for different parameter values')
plt.legend()


# plot FDM solution for different values of N
a = a_list[-1]
plt.figure(2)
plt.plot( grid2, u(grid2,a), label = 'exact')
for N in N_list:
    grid = np.linspace(0,1,N+2)
    plt.plot(grid[1:-1], u_h(a,N),'.-',label='N = %d' % N)
plt.ylim([0,1.4])
plt.xlabel('x')
plt.ylabel('u(x,a), u_h(x,a)')
plt.title('exact solution vs FDM solution for a = %.4f' % a)
plt.legend(loc=8)



plt.show()


