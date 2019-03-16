"""
FEniCS tutorial demo program: 
Heat equation with heat source and Dirichlet b.c.
Adaptation of d1_d2D.py
"""

from dolfin import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def parameters():
    # parameters
    nx = ny = 30 # spatial refinement on x and y axis
    N_k = 80    # number of time steps
    T = 0.1 # final time
    
    return nx, ny, N_k, T

def initialCond():
    # Define initial condition
    code = "0.5 - 1.5*sqrt((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) \
            +fabs(0.5 - 1.5*sqrt((x[0]-0.5)*(x[0]-0.5) + \
            (x[1]-0.5)*(x[1]-0.5)))"

    u0 = Expression(code, degree = 2)
    return u0

def boundaryCond():
    # Define boundary conditions
    u_bc = Constant('0.0')

    return u_bc

def heatSource():
    # Define heat source f

    #code = "10 - 30*sqrt((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) \
    #        +fabs(10 - 30*sqrt((x[0]-0.5)*(x[0]-0.5) + \
    #        (x[1]-0.5)*(x[1]-0.5)))"
    #
    # f = Expression(code, degree = 2)

    f = Expression('100*t', t = 0.0, degree = 2)
    return f

def plot_Function2d(V,mesh,u,fig=None):
    # extract mesh coordinates
    coordinates = mesh.coordinates()
    x_coord = coordinates[:,0]
    y_coord = coordinates[:,1]
    
    # extract values of u on vertices
    vertex_values = u.compute_vertex_values(mesh)

    # surf plot with pyplot
    if fig == None:
        fig = plt.figure() # create figure if none given
    ax = fig.gca(projection='3d')
    ax.cla()
    ax.plot_trisurf(x_coord, y_coord, vertex_values, cmap=plt.cm.hot,vmax=1)
    ax.grid(False)

    return fig

def solveHeatEq(u_ini, u_bc, f, mesh, T, N_k):

    k = T/N_k # temporal step size

    # define function space
    V = FunctionSpace(mesh, 'Lagrange', 1)
    
    def u0_boundary(x, on_boundary):
        return on_boundary
    
    bc = DirichletBC(V, u_bc, u0_boundary)
    
    # set initial condition
    u_1 = interpolate(u_ini, V) # or project(...)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx + k*inner(nabla_grad(u), nabla_grad(v))*dx
    L = (u_1 + k*f)*v*dx

    A = assemble(a)   # assemble A only once before the time stepping
    b = None          # necessary for memory saving assemble call
    
    # plot initial condition
    fig = plt.figure()  
    t = 0.
    fig = plot_Function2d(V,mesh,u_1,fig=fig)
    ax = fig.gca(projection='3d')
    ax.set_title('t = %.2f' % t)
    ax.set_zlim(0,1) 
    plt.pause(2)

    # Compute solution
    u = Function(V)   # the unknown at a new time level
    
    for i in xrange(N_k):
        t = i*k
        f.t = t # update time for heat source
        # assemble right hand side from current L 
        b = assemble(L, tensor=b)
        bc.apply(A, b) # apply boundary conditions to A and b
        solve(A, u.vector(), b) # compute new solution
        # feed new solution into L
        u_1.assign(u)
        # create surf plot of u_1 
        fig = plot_Function2d(V,mesh,u_1,fig=fig)
        ax = fig.gca(projection='3d')
        ax.set_title('t = %.2f' % t)
        ax.set_zlim(0,1) 
        plt.pause(0.05)
        

def main():
    # get parameter values
    nx, ny, N_k, T = parameters()
    # generate mesh 
    mesh = UnitSquareMesh(nx, ny)
    # get initial condition, boundary condition and heat source
    u_ini = initialCond()
    u_bc = boundaryCond()
    f = heatSource()
    # f = Constant('0.0')
    # numerical solution of Heat Eq with animation
    solveHeatEq(u_ini, u_bc, f, mesh, T, N_k)


if __name__ == "__main__":
    main()
