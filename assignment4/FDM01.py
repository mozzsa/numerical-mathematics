"""


A well-behaved FDM example.

How to use: 
On the command line type
python FDM01.py N
for some positive integer N, which determines the number of grid points
Note that
python FDM01.py 
gives the same as
python FDM01.py 10
"""

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import sys

# parameter values
if len(sys.argv) == 1:
    N = 10          # number of interior grid points
else:
    N = int(sys.argv[1]) # or get if from command line

h = 1./(N+1)    # step size
a = 0.5         # diffusion parameter

f = lambda x: x*(1-x) # function handle for inhomogenity 

u = lambda x: x**4/6. - x**3/3. + x/6. # exact solution

grid_ = np.linspace(0,1,N+2)    # uniform grid
grid = grid_[1:-1]              # interior points of the grid

# generate L_h

L_h = -a/h**2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format="csr")

# generate f_h

f_h = f(grid)

# compute FDM solution

u_h = spsolve(L_h, f_h)

# compute the error

err = u_h - u(grid)
err_max = np.linalg.norm(err,np.inf)

# compute the matrix norm of L_h^{-1}:

L_h_inv = np.linalg.inv(L_h.toarray())
L_h_inv_norm = np.linalg.norm(L_h_inv,np.inf)


print 'Number of interior points: N = ', N
print 'Discretization error in max norm: ', err_max 
print 'Matrix norm of inv(L_h) = ', L_h_inv_norm


# plot exact and discrete solutions
grid2 = np.linspace(0,1,4*(N+2)) # finer grid for exact solution
plt.figure(1)
plt.plot(grid2, u(grid2),label='u')
plt.plot(grid, u_h,'.',label='u_h')
plt.xlabel('x')
plt.ylabel('u(x), u_h(x)')
plt.legend()
plt.title('exact vs discrete solution')

# plot the error
plt.figure(2)
plt.plot(grid, np.abs(err),'.',label='error')
plt.xlabel('x')
plt.ylabel('error(x)')
plt.legend()
plt.title('error at the grid points')

plt.show()
