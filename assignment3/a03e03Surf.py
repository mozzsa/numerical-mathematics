
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from a03e03WaveEq import a03e03WaveEq

import numpy as np
import matplotlib.pyplot as plt


def a03e03Surf():
    t = np.linspace(0, 10, 1000)
    x = np.linspace(-15, 15, 1000)
    x, t = np.meshgrid(x, t)
    
    a = 0
    b = 1
    z = a03e03WaveEq(t, x, a, b)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(x, t, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    z_min = np.min(z)
    z_max = np.max(z)
    # Customize the z axis.
    ax.set_title('a=0, b=1')
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('x')
    ax.set_ylabel('time [s]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    a = 1
    b = 0
    z = a03e03WaveEq(t, x, a, b)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(x, t, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    z_min = np.min(z)
    z_max = np.max(z)
    # Customize the z axis.
    ax.set_title('a=1, b=0')
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('x')
    ax.set_ylabel('time [s]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    a = 1
    b = 1
    z = a03e03WaveEq(t, x, a, b)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(x, t, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    z_min = np.min(z)
    z_max = np.max(z)
    # Customize the z axis.
    ax.set_title('a=1, b=1')
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('x')
    ax.set_ylabel('time [s]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()