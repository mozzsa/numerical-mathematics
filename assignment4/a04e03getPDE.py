"""

a)
fh =[f1 - ((-2*a - b*h_i)/(h_i*(h_i + h_i+1)))*alpha,
f2
.
.
.
fN -((b*h_i+1 - 2*a)/(h_i+1*(h_i + h_i+1)))*beta]

If we change the differences only the b therm changes!
So we get:
    D+ -> b(0, -1/h_i+1, 1/h_i+1)
    D- -> b(-1/hi, 1/h_i, 0)

"""

from scipy.sparse import diags
import numpy as np

def a04e03getPDE(xh, f , consts, flag):
    
    #Getting constants from the input vector
    a = consts[0]
    b = consts[1]
    c = consts[2]
    alpha = consts[3]
    beta = consts[4]
    
    #Setting up vector lengths
    h = np.zeros(len(xh)-1)             #all step sizes
    center = np.zeros(len(h)-1)         #Lh entries for uh(x[i])
    left = np.zeros(len(h)-2)           #Lh entries for uh(x[i-1])
    right = np.zeros(len(h)-2)          #Lh entries for uh(x[i+1])

    #calculating all step sizes    
    for i in range(0, len(xh)-1):
        h[i] = xh[i+1] - xh[i]
    
    #checking for flag
    if flag == "0":
        #Calculating all diagonal entries for center, left and right!
        for k in range(0, len(h)-1):
            center[k] = 2.*a/(h[k]*h[k+1]) + c
            
        for j in range(0, len(h)-2):
            left[j] = (-2.*a - b*h[j])/(h[j]*(h[j]+h[j+1]))
            right[j] = (b*h[j+1] - 2.*a)/(h[j+1]*(h[j] + h[j+1]))
        
        Lh = diags([left, center, right],[-1, 0, 1], shape=(len(xh)-2, len(xh)-2))   #calculating Lh as spare Matrix!
        f[0] = f[0] - ((-2.*a - b*h[0])/(h[0]*(h[0] + h[1])))*alpha                  #calculating first value with BC!
        f[-1] = f[-1] - ((b*h[-1] - 2.*a)/(h[-1]*(h[len(h)-2] + h[-1])))*beta        #calculating last value with BC!
        
    elif flag == "-":
        for k in range(0, len(h)-1):
            center[k] = (2.*a + b*h[k+1])/(h[k]*h[k+1]) + c
            
        for j in range(0, len(h)-2):
            left[j] = (-2.*a - b*(h[j] + h[j+1]))/(h[j]*(h[j] + h[j+1]))
            right[j] = -2.*a/(h[j+1]*(h[j] + h[j+1]))
        
        Lh = diags([left, center, right],[-1, 0, 1], shape=(len(xh)-2, len(xh)-2))
        f[0] = f[0] - ((-2.*a - b*(h[0] + h[1]))/(h[0]*(h[0] + h[1])))*alpha
        f[-1] = f[-1] + 2.*a/(h[-1]*(h[len(h)-2] + h[-1]))*beta
        
    elif flag == "+":
        for k in range(0, len(h)-1):
            center[k] = (2.*a - b*h[k])/(h[k]*h[k+1]) + c
            
        for j in range(0, len(h)-2):
            left[j] = -2.*a/(h[j]*(h[j]+h[j+1]))
            right[j] = (b*(h[j] + h[j+1]) - 2.*a)/(h[j+1]*(h[j] + h[j+1]))
        
        Lh = diags([left, center, right],[-1, 0, 1], shape=(len(xh)-2, len(xh)-2))
        f[0] = f[0] + 2.*a/(h[0]*(h[0] + h[1]))*alpha
        f[-1] = f[-1] - (b*(h[len(h)-2] + h[-1]) - 2.*a)/(h[-1]*(h[len(h)-2] + h[-1]))*beta
        
    return Lh, f