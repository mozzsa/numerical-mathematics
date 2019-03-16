

from a03e03WaveEq import a03e03WaveEq
from a08e03LaxWend import a08e03LaxWend
import numpy as np

#Setting up functions
f = lambda x: 1./((x-4)**2 +1) + 1./((x+4)**2 + 1)
g = lambda x: x*np.exp(-1./2*x**2)

#exact solution of wave eq (d'Alembert from exercise a03e03)
a = 1
b = 1
t = 1
x = 2
uex = a03e03WaveEq(t, x, a, b)

#Setting up vectors
N = np.zeros(8)
h = np.zeros(8)
k = np.zeros(8)
error = np.zeros(8)

for i in range(3, 11):
    N[i-3] = 2**i
    h[i-3] = 1./N[i-3]
    k[i-3] = 1./2*h[i-3]

#Setting up for calculating u(1,2)
xmin = 2
xmax = 2
c = 1

#Calculating all Errors
for i in range(0, 8):
    err = a08e03LaxWend(t, xmin, xmax, c, f, g, h[i], k[i]) - uex
    error[i] = np.linalg.norm(err, np.inf)
    
print 'All errors chronologically from p={3,...,10}: \n' , error

#output for errors
#All errors chronologically from p={3,...,10}: 
# [ 0.02224396  0.02253917  0.02248026  0.02239562  0.02233887  0.02230679
#  0.02228981  0.02228108]
# They are nearly same for p={3,...,10}