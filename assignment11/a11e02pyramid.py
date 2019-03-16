"""
Numerical Mathematics for Engineering II WS 17/18
Assignment 11 Programming Exercise 02:
    
Programm creates a surface plot of a basis function on a triangular mesh.
"""

import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def a11e02pyramid(i,j,h,order):
    """Create a lambda function representing the pyramid function or its weak
    gradient (determined by order) centered over the basis node (i*h,j*h) of a
    triangular mesh with step size h."""
    
    # support of the reference pyramid function over [-1,1]^2 numbering counter
    # clockwise
    """
                     -----
                    /| 2 /|
                   / |  / |
                  /  | /  |
                 / 3 |/ 1 |
                 ----------
                | 4 /| 6 /
                |  / |  /
                | /  | /
                |/ 5 |/
                ------
    """
    supp_tri_1 = lambda x,y: (x<=1) * (0<=y) * (y<=x)
    supp_tri_2 = lambda x,y: (0<x) * (y<1) * (y>x)
    supp_tri_3 = lambda x,y: (x<=0) * (0<=y) * (y<=x+1) * (x!=y)    # exclude center (0,0)
    supp_tri_4 = lambda x,y: (-1<x) * (y<0) * (y>x)
    supp_tri_5 = lambda x,y: (x<=0) * (-1<=y) * (y<=x) * (x!=-y)    # exclude center (0,0)
    supp_tri_6 = lambda x,y: (0<x) * (y<0) * (y>x-1)


    # Case: pyramid function \phi^h_{i,j}
    if(order==0):
        
        # reference pyramid function
        ref_phi = lambda x,y: (1-x) * supp_tri_1(x,y) \
                                + (1-y) * supp_tri_2(x,y) \
                                + (1+x-y) * supp_tri_3(x,y) \
                                + (1+x) * supp_tri_4(x,y) \
                                + (1+y) * supp_tri_5(x,y) \
                                + (1-x+y) * supp_tri_6(x,y)
                                
        # pyramid function (by applying coordinate transformation from original
        # to reference support)
        func = lambda x,y: ref_phi(x/h-i,y/h-j)
                     
    # Case: weak gradient of pyramid function \nabla \phi^h_{i,j}
    elif(order==1):
        
        # first entry of weak gradient for reference pyramid function
        ref_grad_phi_1 = lambda x,y: (-1) * (supp_tri_1(x,y) + supp_tri_6(x,y)) \
                                      + 1 * (supp_tri_3(x,y) + supp_tri_4(x,y))
        # second entry of weak gradient for reference pyramid function
        ref_grad_phi_2 = lambda x,y: (-1) * (supp_tri_2(x,y) + supp_tri_3(x,y)) \
                                      + 1 * (supp_tri_5(x,y) + supp_tri_6(x,y))
                                
        # weak gradient of pyramid function (by applying coordinate
        # transformation from original to reference support)
        func = lambda x,y: [ 1./h * ref_grad_phi_1(x/h-i,y/h-j), \
                             1./h * ref_grad_phi_2(x/h-i,y/h-j)]
    
    # Return pyramid function or its weak gradient
    return func

def a11e02surface_plot():
    """Surface plot of the pyramid function."""
    
    # step size for triangular mesh
    h = 1./4
    
    # Set basis node for pyramid function
    i = 1
    j = 1
    
    # Generate mesh for plotting
    points = 250
    grid_1D = np.linspace(0,1,points)
    X,Y = np.meshgrid(grid_1D,grid_1D)
    
    # Evaluate pyramid function related to the basis node (i*h,j*h)
    phi = a11e02pyramid(i,j,h,0)
    Z = phi(X,Y)
    
    # Create surface plot for pyramid function
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('pyramid function over node (%.2f,%.2f) \n with step size h = %.2f' % (i*h,j*h,h))
    
    # Show plot
    plt.show()
    
def a11e02matrixelement(i,j,k,l,h,order):
    """Computes entries of the mass matrix or stiffness matrix (determined by
    order) for the basis functions related to the nodes (i*h,j*h) and (k*h,l*h)
    of a triangular mesh with step size h."""
    
    # Case: approximation of \langle \phi_{ij}^h,\phi_{kl}^h \rangle_{L^2(\Omega)}
    if(order==0):
        
        # Get basis functions
        phi_ij = a11e02pyramid(i,j,h,0)
        phi_kl = a11e02pyramid(k,l,h,0)
        # production of basis functions
        func = lambda x,y: phi_ij(x,y) * phi_kl(x,y)
    
    # Case: approximation of a(\phi_{ij}^h,\phi_{kl}^h)
    elif(order==1):
        
        # Get weak gradients of basis functions
        phi_d_ij = a11e02pyramid(i,j,h,1)
        phi_d_kl = a11e02pyramid(k,l,h,1)
        # inner product of weak gradients of basis functions
        func = lambda x,y: phi_d_ij(x,y)[0] * phi_d_kl(x,y)[0] + phi_d_ij(x,y)[1] * phi_d_kl(x,y)[1]
    
    # Set integration bounds of inner intergal for quadrature
    bd0 = lambda y: 0
    bd1 = lambda y: 1
    
    # Calculate desired approximation using quadrature in 2D
    approx, err = dblquad(func,0,1,bd0,bd1)
    
    return approx

if __name__ == "__main__":
    a11e02surface_plot()