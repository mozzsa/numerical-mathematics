"""


Some FD methods for the wave equation.

How to use: 
On the command line type
python FDM04.py arg1 arg2

arg1 specifies the scheme:
    1 Friedrichs method (default)
    2 Lax Wendroff method

arg2 specifies the number of time steps N_k
    default value N_k = 300

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

W = []

def Friedrichs_Vec(t,jmin,jmax,A,v,h,k):
    """
    The Friedrichs scheme for the vector valued transport equation.
    """
    # dimension
    d = A.shape[1]
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
        U = 0.5*np.dot(np.eye(d) + lam*A,U[:,2:]) + 0.5*np.dot( np.eye(d)
                - lam*A,U[:,:-2])
        ell = ell - 1
    return U

def LaxWendroff_Vec(t,jmin,jmax,A,v,h,k):
    """
    The Lax-Wendroff scheme for the transport equation.
    Convergent of order 2 w.r.t. discrete L2 norm.
    """
    # dimension
    d = A.shape[1]
    # ratio between h and k
    lam = k/1./h
    # number of time steps
    ell = int(np.floor(t/1./k))
    # generate grid 
    x_grid = h*np.arange(jmin-ell,jmax+ell+1)
    # evaluate initial condition 
    U = v(x_grid)
    # recursion for the Lax-Wendroff scheme
    coef1 = np.dot(A,A)*lam**2
    coef2 = A*lam
    while ell > 0:
        U = np.dot(np.eye(d) - coef1, U[:,1:-1]) + 0.5*np.dot(coef1 + coef2, 
                U[:,2:]) + 0.5*np.dot( coef1 - coef2, U[:,:-2])
        ell = ell - 1
    return U


# initial conditions plus their (anti-)derivatives
def v1(x):
    return 1./(1 + (x-3)**2) + 1./ (1 + (x + 3)**2)

def Dv1(x):
    return -2.*(x-3)/(1 + (x-3)**2)**2 - 2.*(x+3)/ (1 + (x + 3)**2)**2

def v2(x):
    return x*np.exp(-0.5*x**2)

def V2(x):
    return -1*np.exp(-0.5*x**2)

def exact_sol(t,x,a):
    return (0.5*( v1(x - a*t) + v1(x + a*t) )
            + 0.5/a*( V2(x +a*t) - V2(x -a*t)) )

# initial condition as needed for the vectorized scheme
def v2d(x,a):
    U = np.zeros((2,len(x)))
    U[0,:] = a*Dv1(x)
    U[1,:] = v2(x)
    return U

def animate(t,x_min,x_max,a,h,k,x_grid,line,line2,M):
    # update exact solution
    line.set_ydata(exact_sol(t,x_grid,a));

    # update numerical solution
    jmax = int(np.floor(x_max/1./h))
    jmin = int(np.ceil(x_min/1./h))
    x_grid2 = h*np.arange(jmin,jmax+1)

    A = np.array([[0., a],[a, 0.]])

    v2d_aux = lambda (x): v2d(x,a)

    if M == 1:
        U = Friedrichs_Vec(t,jmin,jmax,A,v2d_aux,h,k)
        str_title = 'Friedrichs method, N_k = %d' % N_k
    elif M == 2:
        U = LaxWendroff_Vec(t,jmin,jmax,A,v2d_aux,h,k)
        str_title = 'Lax Wendroff method, N_k = %d' % N_k

    # Attention: !!!Using a global variable is not good programming practice!!! 
    global W
    if t <= 0:
        W = v1(x_grid2)
    else:
        W = W + k*U[1,:]
    
    line2.set_ydata(W)


def main(M, N_k = 300):
    """
    Main function
    """
    
    # list of parameter values
    x_min = -15
    x_max = 15

    a = 1.
    T = 10.
    
    k = T/1./N_k
    h = 2*a*k
    
    global W

    # plot exact solution
    x_grid = np.linspace(x_min,x_max,N_k + 1)

    fig, ax = plt.subplots()

    line, = ax.plot(x_grid, v1(x_grid))

    # compute initial condition of num_scheme
    jmax = int(np.floor(x_max/1./h))
    jmin = int(np.ceil(x_min/1./h))
    x_grid2 = h*np.arange(jmin,jmax+1)

    W = v1(x_grid2) 
    line2, = ax.plot(x_grid2, W, '.g')

    if M == 1:
        str_title = 'Friedrichs method, N_k = %d' % N_k
    elif M == 2:
        str_title = 'Lax Wendroff method, N_k = %d' % N_k

    ax.set_title(str_title)
    ax.set_ylim([-0.5, 1.2])

    # function handle for animation
    animate_aux = lambda(t): animate(t,x_min,x_max,a,h,k,x_grid,line,line2,M)
        
    ani = animation.FuncAnimation(fig, animate_aux, np.linspace(0,T,N_k+1),
            interval = 25,repeat=False)
    ani.save('FDM04_scheme_%d_N_k_%d.avi' % (M, N_k))
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



