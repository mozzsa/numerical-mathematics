

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from a03e03WaveEq import a03e03WaveEq


def a03e03Mov(a,b): 
   fig, ax = plt.subplots()
   ax.set_xlim(( -15, 15))
   ax.set_ylim((-0.5,1.3))
   line, = ax.plot([], [], lw=2)
   line, = init(line,)
   t = np.linspace(0,10,300)
   x = np.linspace(-15, 15, 1000)
   anim = animation.FuncAnimation(fig, animate,fargs=(a,b,t,x,line,),
                               frames=300, interval=20, blit=True)
   anim.save('animation_wave.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 
    

def init(line,):
    line.set_data([], [])
    return (line,)

def animate(i,a,b,t,x,line,):
    y = a03e03WaveEq(t[i],x, a, b)
    
    line.set_data(x, y)
    return (line,)

    










