"""


A test problem for linear solvers

How to use: 
On the command line type
python LinAlg01.py

optional command line arguments
n = dimension of test problem
tol = relative tolerance

"""

import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import sys

def Jacobi(A,b,x_0,u_hat,tol):
    A = A.copy()
    norm_u = np.linalg.norm(u_hat, 2)
    rel_err = np.linalg.norm(u_hat - x_0, 2)/ norm_u
    it = 0
    err = np.array(rel_err)
    n = x_0.size     # catch dimension of the system

    diag = A.diagonal() # extract diagonal of A
    A.setdiag(np.zeros(n)) # set diagonal to 0 in A
    A.eliminate_zeros() # eliminate zeros from A

    while rel_err > tol:
        it = it + 1 # increase counter for iterations

        x_new = (b - A.dot(x_0))/diag # compute M^{-1} (b - N x_old)
        # prepare for next iteration, compute error
        x_0 = x_new.copy() 
        rel_err = np.linalg.norm(u_hat - x_0, 2)/ norm_u
        err = np.append(err,rel_err)
    return x_0, it, err


def GaussSeidel(A,b,x_0,u_hat,tol):
    A = A.copy()
    x = x_0.copy()
    norm_u = np.linalg.norm(u_hat, 2)
    rel_err = np.linalg.norm(u_hat - x_0, 2)/ norm_u
    it = 0
    err = np.array(rel_err)
    n = x_0.size  # catch dimension of the system
    
    diag = A.diagonal() # extract diagonal of A
    A.setdiag(np.zeros(n)) # set diagonal to 0 in A
    A.eliminate_zeros() # eliminate zeros from A
    
    # extract rows of A 
    row = []
    for i in xrange(n):
        row.append(A.getrow(i))

    while rel_err > tol:
        for i in xrange(n):
            x[i] = (b[i] - row[i].dot(x) )/diag[i]
        # prepare for next iteration, compute error
        it = it + 1 # increase counter for iterations
        rel_err = np.linalg.norm(u_hat - x, 2)/ norm_u
        err = np.append(err,rel_err)
    return x, it, err

def SymGaussSeidel(A,b,x_0,u_hat,tol):
    A = A.copy()
    x = x_0.copy()
    norm_u = np.linalg.norm(u_hat, 2)
    rel_err = np.linalg.norm(u_hat - x_0, 2)/ norm_u
    it = 0
    err = np.array(rel_err)
    n = x_0.size  # catch dimension of the system
    
    diag = A.diagonal() # extract diagonal of A
    R = sp.triu(A, k=1)
    L = sp.tril(A, k=-1)

    # extract rows of R and L 
    row_R = []
    row_L = []
    for i in xrange(n):
        row_R.append(R.getrow(i))
        row_L.append(L.getrow(i))
    
    # auxiliary vector
    v = R.dot(x_0)
    w = np.zeros(n)

    while rel_err > tol:
        for i in xrange(n):
            w[i] = row_L[i].dot(x)
            x[i] = ( b[i] - v[i] - w[i])/diag[i]
        for i in xrange(n,0,-1):
            v[i-1] = row_R[i-1].dot(x)
            x[i-1] = ( b[i-1] - v[i-1] - w[i-1])/diag[i-1]
        # prepare for next iteration, compute error
        it = it + 1 # increase counter for iterations
        rel_err = np.linalg.norm(u_hat - x, 2)/ norm_u
        err = np.append(err,rel_err)
    return x, it, err
  
def CGmeth(A,b,x_0,u_hat,tol):
    A = A.copy()
    x = x_0.copy()
    norm_u = np.linalg.norm(u_hat, 2)
    rel_err = np.linalg.norm(u_hat - x_0, 2)/ norm_u
    it = 0
    err = np.array(rel_err)

    # compute residual
    r = b - A.dot(x)
    norm_r = np.linalg.norm(r, 2)**2
    d = r   # initial descent direction

    while rel_err > tol:
        Ad = A.dot(d)
        alpha = norm_r / np.dot(d, Ad)
        x = x + alpha*d
        r = r - alpha*Ad
        norm_r_new = np.linalg.norm(r,2)**2
        beta = norm_r_new/norm_r
        norm_r = norm_r_new
        d = r + beta*d
        # prepare for next iteration, compute error
        it = it + 1 # increase counter for iterations
        rel_err = np.linalg.norm(u_hat - x, 2)/ norm_u
        err = np.append(err,rel_err)
    return x, it, err



def main(n,tol):
    # creating FDM stiffness matrix for elliptic BVP on [0,1] with 0-b.c.
    h = 1./(n+1)
    A = -1/h**2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr")
    b = np.ones(n) # load vector

    # exact solution 
    x_grid = np.linspace(0,1,n+2)
    x_grid = x_grid[1:-1]
    u_exact = lambda(x): x*(1-x)/2.
    u_hat = u_exact(x_grid)

    x_0 = np.zeros(n)
    
    # Jacobi method
    u_J,it_J,err_J = Jacobi(A,b,x_0,u_hat,tol)

    # Gauss Seidel method
    u_GS,it_GS,err_GS = GaussSeidel(A,b,x_0,u_hat,tol)

    # Sym. Gauss Seidel method
    u_SGS,it_SGS,err_SGS = SymGaussSeidel(A,b,x_0,u_hat,tol)

    # CG method
    u_CG,it_CG,err_CG = CGmeth(A,b,x_0,u_hat,tol)
    
    print 'Jacobi : ', it_J, 'Error ', np.linalg.norm(u_hat - u_J, 2)
    print 'Gauss-S: ', it_GS, 'Error ', np.linalg.norm(u_hat - u_GS, 2)
    print 'sym GS : ', it_SGS, 'Error ', np.linalg.norm(u_hat - u_SGS, 2)
    print 'CG meth: ', it_CG, 'Error ', np.linalg.norm(u_hat - u_CG, 2)

    plt.semilogy(np.arange(it_J+1),err_J,label='Jacobi')
    plt.semilogy(np.arange(it_GS+1),err_GS,label='Gauss-Seidel')
    plt.semilogy(np.arange(it_SGS+1),err_SGS,label='sym. Gauss-Seidel')
    plt.semilogy(np.arange(it_CG+1),err_CG,label='CG method')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('relative error')

    plt.ylim((10**-5,1))

    plt.show()

    

if __name__ == "__main__":
    n = 100         
    tol = 10**(-4)  
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        tol = float(sys.argv[1])

    main(n,tol)



