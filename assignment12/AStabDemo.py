""" Demo: A-stability for the temporal discretization of the heat equation """

import numpy as np
from scipy.fftpack import dst
import matplotlib.pyplot as plt

# We consider full discretizations of the heat equation on [0,1] with zero
# Dirichlet boundary conditions. The spatial discretization is done by a finite
# difference scheme with step size h. The temporal discretization is carried
# out by the forward and backward Euler method.

# switches between the options
# opt == 0: just the exact solution
# opt == 1: backward Euler + exact solution
# opt == 2: forward Euler + exact solution

opt = 2

# initial condition
def u_0(x):
    return 6*np.minimum( np.maximum(0,x-1./3), np.maximum(0, 2./3 - x))

# normalized version of discrete sine transform
def dst_norm(x):
    return dst(x,type=1)/np.sqrt((len(x) + 1.)*2.)


# eigenvalues of Laplace operator
def eigval(n):
    return -np.pi**2*n**2

def exact_sol(t,u_0):
    # returns the exact solution at time t 
    # input: vector of initial condition 
    # returns: vector of exact solution with same resolution as input

    # discrete sine transform of u_val for exact solution
    u_0_coef = dst_norm(u_0[1:-1])

    exact = dst_norm( np.exp(t*eigval(np.arange(1,len(u_0_coef)+1)))*u_0_coef)
    return np.concatenate((np.zeros(1),exact,np.zeros(1)))

def stiffmat(N_h):
    # returns the stiffness matrix 
    h = 1./(N_h + 1)

    A = -2*np.eye(N_h) + np.eye(N_h, k = 1) + np.eye(N_h, k=-1)
    return A/h**2

def forwardEuler(A,k,u_0):
    # returns a finite difference approximation of the heat equation
    # with one step of the forward Euler with step size k
    # input: A stiffness matrix, k step size, u_0 initial value

    U_old = u_0[1:-1]
    U_new = U_old + k*np.dot(A,U_old)

    return np.concatenate((np.zeros(1),U_new,np.zeros(1)))

def backwardEuler(A,k,u_0):
    # returns a finite difference approximation of the heat equation
    # with one step of the forward Euler with step size k
    # input: A stiffness matrix, k step size, u_0 initial value

    U_old = u_0[1:-1]
    U_new = np.linalg.solve((np.eye(len(U_old)) - k*A),U_old)

    return np.concatenate((np.zeros(1),U_new,np.zeros(1)))

def animation_exact(N_h,N_k,T):
    # temporal step size
    k = T/1./N_k
    # spatial grid
    x_grid = np.linspace(0,1,N_h + 2)

    # evaluating initial condition at x_grid
    u_val = u_0(x_grid)

    plt.ion()

    for i in np.arange(N_k+1):
        t = i*k
        if i == 0:
            plt.plot(x_grid,u_val)
            plt.ylabel('u(t,x)')
            plt.xlabel('x')
            plt.title('Heat equation at t = %2.3f' % t) 
            plt.pause(1.)
        else:
            plt.cla()
            plt.plot(x_grid,exact_sol(t,u_val))
            plt.ylabel('u(t,x)')
            plt.xlabel('x')
            plt.ylim(0,1)
            plt.title('Heat equation at t = %2.3f' % t) 
            plt.pause(0.1)

def animation_forward(N_h,N_k,T):
    # temporal step size
    k = T/1./N_k
    # spatial grid
    x_grid = np.linspace(0,1,N_h + 2)

    # evaluating initial condition at x_grid
    u_ini = u_0(x_grid)
    # generate stiffness matrix
    A = stiffmat(N_h)

    plt.ion()

    for i in xrange(N_k+1):
        t = i*k
        if i == 0:
            u_val = u_ini
            plt.plot(x_grid,u_val)
            plt.ylabel('u_hk(t,x)')
            plt.xlabel('x')
            plt.title('forward Euler at t = %2.3f' % t) 
            plt.pause(1.)
        else:
            u_val = forwardEuler(A,k,u_val)
            plt.cla()
            plt.plot(x_grid,u_val)
            plt.plot(x_grid,exact_sol(t,u_ini))
            plt.ylabel('u_hk(t,x)')
            plt.xlabel('x')
            plt.ylim(0,1)
            plt.title('forward Euler at t = %2.3f' % t) 
            plt.pause(0.1)

def animation_backward(N_h,N_k,T):
    # temporal step size
    k = T/1./N_k
    # spatial grid
    x_grid = np.linspace(0,1,N_h + 2)

    # evaluating initial condition at x_grid
    u_ini = u_0(x_grid)
    # generate stiffness matrix
    A = stiffmat(N_h)

    plt.ion()

    for i in xrange(N_k+1):
        t = i*k
        if i == 0:
            u_val = u_ini
            plt.plot(x_grid,u_val)
            plt.ylabel('u_hk(t,x)')
            plt.xlabel('x')
            plt.title('backward Euler at t = %2.3f' % t) 
            plt.pause(1.)
        else:
            u_val = backwardEuler(A,k,u_val)
            plt.cla()
            plt.plot(x_grid,u_val)
            plt.plot(x_grid,exact_sol(t,u_ini))
            plt.ylabel('u_hk(t,x)')
            plt.xlabel('x')
            plt.ylim(0,1)
            plt.title('backward Euler at t = %2.3f' % t) 
            plt.pause(0.1)


def main(opt):
    T = 0.1 # final time
    if opt == 0:
        # parameter values
        N_h = 99
        N_k = 80

        animation_exact(N_h,N_k,T)
    if opt == 1:
        # parameter values
        N_h = 99
        N_k = 80

        animation_backward(N_h,N_k,T)
    if opt == 2:
        # parameter values 
        N_h = 99
        N_k = 160

        animation_forward(N_h,N_k,T)


if __name__ == "__main__":
    main(opt)
