

from a05ex04shishkin import a05ex04shishkin
from a05ex04solve import a05ex04solve
from a05ex04error import a05ex04error
import numpy as np
import matplotlib.pyplot as plt

def a05ex04experiment():
    eps = 0.001
    N = np.array([5, 50, 500, 5000])
    h = 1./(2*N)        #all uniform stepsizes for all N
    
    #setting up exact solution for 1000000 steps between 0 and 1
    u = lambda x: x - (np.exp(-(1.-x)/eps) - np.exp(-1./eps))/(1 - np.exp(-1./eps))
    xh_ex = np.linspace(0, 1, 1000000)
    uex = u(xh_ex)
    
    #Setting up all uniform step sizes for all N
    xh_5 = np.linspace(0, 1, N[0]*2)
    xh_50 = np.linspace(0, 1, N[1]*2)
    xh_500 = np.linspace(0, 1, N[2]*2)
    xh_5000 = np.linspace(0, 1, N[3]*2)
    
    #Setting up all shishkin grids for all N
    xh_shish_5 = a05ex04shishkin(N[0], eps)
    xh_shish_50 = a05ex04shishkin(N[1], eps)
    xh_shish_500 = a05ex04shishkin(N[2], eps)
    xh_shish_5000 = a05ex04shishkin(N[3], eps)
    
    #Setting up for uniform grid and D+
    flag = "+"
    uh_5 = a05ex04solve(eps, xh_5, flag)
    uh_50 = a05ex04solve(eps, xh_50, flag)
    uh_500 = a05ex04solve(eps, xh_500, flag)
    uh_5000 = a05ex04solve(eps, xh_5000, flag)
    
    [err_5, uex_5] = a05ex04error(eps, xh_5, uh_5)
    [err_50, uex_50] = a05ex04error(eps, xh_50, uh_50)
    [err_500, uex_500] = a05ex04error(eps, xh_500, uh_500)
    [err_5000, uex_5000] = a05ex04error(eps, xh_5000, uh_5000)
    
    error = np.array([err_5, err_50, err_500, err_5000])
    print 'Errors for N={5, 50, 500, 5000} of plot 1 (chronologically): \n' , error
    print '\n'
    
    #Setting up only interior points!
    xh_5r = xh_5[1:-1]
    xh_50r = xh_50[1:-1]
    xh_500r = xh_500[1:-1]
    xh_5000r = xh_5000[1:-1]
    
    
    #Plotting for uniform grids and D+ operator
    plt.figure(1)    
    plt.plot(xh_5r, uh_5,'g--', xh_50r, uh_50,'b:', xh_500r, uh_500,'y-.', xh_5000r, uh_5000,'r:', xh_ex, uex,'k:')
    plt.title('Uniform grids, D+ difference operator, N=5 (green), N=50 (blue), N=500 (yellow), N=5000(red), exact solution (black)')
    plt.axis([0, 1, -1, 1])
    
    #Setting up for uniform grids and D0 operator
    flag = "0"
    uh_5 = a05ex04solve(eps, xh_5, flag)
    uh_50 = a05ex04solve(eps, xh_50, flag)
    uh_500 = a05ex04solve(eps, xh_500, flag)
    uh_5000 = a05ex04solve(eps, xh_5000, flag)
    
    [err_5, uex_5] = a05ex04error(eps, xh_5, uh_5)
    [err_50, uex_50] = a05ex04error(eps, xh_50, uh_50)
    [err_500, uex_500] = a05ex04error(eps, xh_500, uh_500)
    [err_5000, uex_5000] = a05ex04error(eps, xh_5000, uh_5000)
    
    error = np.array([err_5, err_50, err_500, err_5000])
    print 'Errors for N={5, 50, 500, 5000} of plot 2 (chronologically): \n' , error
    print '\n'
    
    #Plotting for uniform grids and D0 operator
    plt.figure(2)    
    plt.plot(xh_5r, uh_5,'g--', xh_50r, uh_50,'b:', xh_500r, uh_500,'y-.', xh_5000r, uh_5000,'r:', xh_ex, uex,'k:')
    plt.title('Uniform grids, D0 difference operator, N=5 (green), N=50 (blue), N=500 (yellow), N=5000(red), exact solution (black)')
    plt.axis([0, 1, -1, 1])
    
    
    #Setting up for uniform grids and D- operator
    flag = "-"
    uh_5 = a05ex04solve(eps, xh_5, flag)
    uh_50 = a05ex04solve(eps, xh_50, flag)
    uh_500 = a05ex04solve(eps, xh_500, flag)
    uh_5000 = a05ex04solve(eps, xh_5000, flag)
    
    [err_5, uex_5] = a05ex04error(eps, xh_5, uh_5)
    [err_50, uex_50] = a05ex04error(eps, xh_50, uh_50)
    [err_500, uex_500] = a05ex04error(eps, xh_500, uh_500)
    [err_5000, uex_5000] = a05ex04error(eps, xh_5000, uh_5000)
    
    error = np.array([err_5, err_50, err_500, err_5000])
    print 'Errors for N={5, 50, 500, 5000} of plot 3 (chronologically): \n' , error
    print '\n'
    
    #Plotting for uniform grids and D- operator
    plt.figure(3)    
    plt.plot(xh_5r, uh_5,'g--', xh_50r, uh_50,'b:', xh_500r, uh_500,'y-.', xh_5000r, uh_5000,'r:', xh_ex, uex,'k:')
    plt.title('Uniform grids, D- difference operator, N=5 (green), N=50 (blue), N=500 (yellow), N=5000(red), exact solution (black)')
    plt.axis([0, 1, -1, 1])
    
    
    #Setting up for shishkin grids and D0 operator
    flag = "0"
    uh_5 = a05ex04solve(eps, xh_shish_5, flag)
    uh_50 = a05ex04solve(eps, xh_shish_50, flag)
    uh_500 = a05ex04solve(eps, xh_shish_500, flag)
    uh_5000 = a05ex04solve(eps, xh_shish_5000, flag)
    
    [err_5, uex_5] = a05ex04error(eps, xh_shish_5, uh_5)
    [err_50, uex_50] = a05ex04error(eps, xh_shish_50, uh_50)
    [err_500, uex_500] = a05ex04error(eps, xh_shish_500, uh_500)
    [err_5000, uex_5000] = a05ex04error(eps, xh_shish_5000, uh_5000)
    
    error = np.array([err_5, err_50, err_500, err_5000])
    print 'Errors for N={5, 50, 500, 5000} of plot 4 (chronologically): \n' , error
    
    #Setting up only interior points!
    xh_shish_5 =  xh_shish_5[1:-1]
    xh_shish_50 =  xh_shish_50[1:-1]
    xh_shish_500 = xh_shish_500[1:-1]
    xh_shish_5000 = xh_shish_5000[1:-1]
    
    #Plotting for shishkin grids and D0 operator
    plt.figure(4)    
    plt.plot(xh_shish_5, uh_5,'g--', xh_shish_50, uh_50,'b:', xh_shish_500, uh_500,'y-.', xh_shish_5000, uh_5000,'r:', xh_ex, uex,'k:')
    plt.title('Shishkin grids, D0 difference operator, N=5 (green), N=50 (blue), N=500 (yellow), N=5000(red), exact solution (black)')
    plt.axis([0, 1, -1, 1])