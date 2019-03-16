#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from a08e03LaxWend import a08e03LaxWend
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np 

c = 1
xmin=-15
xmax = 15
f = lambda x: 1./((x-4)**2 +1) + 1./((x+4)**2 + 1)
g = lambda x: x*np.exp(-1./2*x**2)
N = 2**6
h = 1.0 /N
k = 1./2*h
tt = int(10.0/k)+1
x = np.linspace(xmin,xmax,(xmax-xmin)/h+1)
t = np.linspace(0,10,tt)

def init(line,):
    line.set_data([], [])
    return (line,)

def animate(i,t,x,xmin,xmax,c,f,g,h,k,ax,line,):
    y = a08e03LaxWend(t[i],xmin,xmax,c,f,g,h,k)
    ax.set_ylim((min(y),max(y)))
    line.set_data(x, y)
    return (line,)

fig, ax = plt.subplots()
ax.set_xlim(( -15, 15))
line, = ax.plot([], [], lw=2)
line, = init(line,)
anim = animation.FuncAnimation(fig, animate,fargs=(t,x,xmin,xmax,c,f,g,h,k,ax,line,),
                               frames=tt, interval=11, blit=True)
anim.save('animation_wave_Lax5.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 