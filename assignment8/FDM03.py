"""


Some FD methods for the transport equation.

How to use: 
On the command line type
python FDM03.py arg1 arg2

arg1 specifies the scheme:
    1 Upwind forward (default)
    2 Upwind backward
    3 Central difference (Unstable!)
    4 Friedrichs method
    5 Lax Wendroff method

arg2 specifies the number of time steps N_k
    default value N_k = 300
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


def upwind_for(t,jmin,jmax,a,v,h,k):
    """
    The upwind scheme with forward difference in space.
    Suitable for transport equation with a > 0.
    """

    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin,jmax+ell+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for upwind scheme
    while ell > 0:
        U = a*lam*U[1:] + (1. - a*lam)*U[:-1]
        ell = ell - 1
    return U

def upwind_back(t,jmin,jmax,a,v,h,k):
    """
    The upwind scheme with backward difference in space.
    Suitable for transport equation with a < 0.
    """
    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin-ell,jmax+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for upwind scheme
    while ell > 0:
        U = -a*lam*U[:-1] + (1. + a*lam)*U[1:] 
        ell = ell - 1
    return U



def Friedrichs(t,jmin,jmax,a,v,h,k):
    """
    The Friedrichs scheme for the transport equation.
    """
    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin-ell,jmax+ell+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for the Friedrichs scheme
    while ell > 0:
        U = 0.5*(1. + a*lam)*U[2:] + 0.5*(1. - a*lam)*U[:-2]
        ell = ell - 1
    return U


        
def central(t,jmin,jmax,a,v,h,k):
    """
    The central difference scheme for the transport equation.
    This method is _unstable_!
    """

    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin-ell,jmax+ell+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for the central difference scheme
    while ell > 0:
        U = U[1:-1] + 0.5*a*lam*( U[2:] - U[:-2] )
        ell = ell - 1
    return U


def LaxWendroff(t,jmin,jmax,a,v,h,k):
    """
    The Lax-Wendroff scheme for the transport equation.
    Convergent of order 2 w.r.t. discrete L2 norm.
    """

    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin-ell,jmax+ell+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for the Lax-Wendroff scheme
    coef1 = a**2*lam**2
    coef2 = a*lam
    while ell > 0:
        U = (1 -  coef1)*U[1:-1] + 0.5*(coef1 + coef2)*U[2:] + 0.5*(
                coef1 - coef2)*U[:-2]
        ell = ell - 1
    return U



def v(x):
    """
    The initial condition
    Mexican hat wavelet
    """
    return 2/np.sqrt(3 * np.sqrt(np.pi))* ( 1 - x**2)*np.exp(-0.5*x**2)

def animate(t,x_min,x_max,a,v,h,k,x_grid,line,line2,M):
    line.set_ydata(v(x_grid + a * t));
    jmax = int(np.floor(x_max/1./h))
    jmin = int(np.ceil(x_min/1./h))
    if M == 1:
        U = upwind_for(t,jmin,jmax,a,v,h,k)
    elif M == 2:
        U = upwind_back(t,jmin,jmax,a,v,h,k)
    elif M == 3:
        U = central(t,jmin,jmax,a,v,h,k)
    elif M == 4:
        U = Friedrichs(t,jmin,jmax,a,v,h,k)
    elif M == 5:
        U = LaxWendroff(t,jmin,jmax,a,v,h,k)
    line2.set_ydata(U)



def main(M, N_k = 200):
    """
    Main function
    """
    
    # list of parameter values
    x_min = -6
    x_max = 4

    a = 1
    T = 6.
    
    k = T/1./N_k
    h = 2*a*k

    # plot exact solution
    x_grid = np.linspace(x_min,x_max,N_k + 1)

    fig, ax = plt.subplots()

    line, = ax.plot(x_grid, v(x_grid))

    # compute numerical approximation
    jmax = int(np.floor(x_max/1./h))
    jmin = int(np.ceil(x_min/1./h))

    if M == 1:
        U = upwind_for(0,jmin,jmax,a,v,h,k)
        str_title = 'Upwind forward scheme, N_k = %d' % N_k
    elif M == 2:
        U = upwind_back(0,jmin,jmax,a,v,h,k)
        str_title = 'Upwind backward scheme, N_k = %d' % N_k
    elif M == 3:
        U = central(0,jmin,jmax,a,v,h,k)
        str_title = 'Central difference, N_k = %d' % N_k
    elif M == 4:
        U = Friedrichs(0,jmin,jmax,a,v,h,k)
        str_title = 'Friedrichs method, N_k = %d' % N_k
    elif M == 5:
        U = LaxWendroff(0,jmin,jmax,a,v,h,k)
        str_title = 'Lax Wendroff method, N_k = %d' % N_k

    # plot numerical solution in same axis
    x_grid2 = h*np.arange(jmin, jmax + 1)
    line2, = ax.plot(x_grid2, U, '.g')
    ax.set_title(str_title)

    # function handle for animation
    animate_aux = lambda(t): animate(t,x_min,x_max,a,v,h,k,x_grid,line,line2,M)
        
    ani = animation.FuncAnimation(fig, animate_aux, np.linspace(0,T,N_k + 1),
            interval = 25)
    ani.save('FDM03_scheme_%d_N_k_%d.avi' % (M, N_k))
    plt.show()


if __name__ == "__main__":
    N_k = 300
    M = 1
    if len(sys.argv) == 2:
        M = int(sys.argv[1])
    elif len(sys.argv) == 3:
        M = int(sys.argv[1])
        N_k = int(sys.argv[2])
        
    main(M,N_k)



