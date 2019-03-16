

import numpy as np

def a03e03WaveEq(t, x, a, b):
    c = 1
    return 0.5*a*(1./((x+c*t-4)**2 + 1) + 1./((x+c*t+4)**2 + 1) \
                  + 1./((x-c*t-4)**2 + 1) + 1./((x-c*t+4)**2 + 1)) \
                  + 0.5*c*(b*np.exp(-0.5*(x-c*t)**2) - b*np.exp(-0.5*(x+c*t)**2))