"""
FEniCS tutorial demo program: 
Heat equation with heat source and Dirichlet b.c.,
Output is an estimate of the experimental order of convergence
Adaptation of d1_d2D.py
"""

from dolfin import *
import numpy as np

def parameters():
    # parameters
    lam = 0.1
    T = 0.1 # final time
    
    return lam, T

def boundaryCond():
    # Define boundary conditions
    u_bc = Constant('0.0')

    return u_bc

def heatSource(lam):
    # Define heat source f
    code = "-16*lam*(1-x[0])*x[0]*(1-x[1])*x[1]*exp(-lam*t) + \
            32*exp(-lam*t)*( (1-x[0])*x[0] + (1-x[1])*x[1] )"
    
    f = Expression(code, t=0.0, lam=lam, degree=4)
 
    return f

def exact_sol(lam):
    # Define exact solution
    code = "16*(1-x[0])*x[0]*(1-x[1])*x[1]*exp(-lam*t)"
    u_e = Expression(code, t=0.0, lam=lam, degree=4)
    return u_e

def L2ErrHeatEq(u_e, u_ini, u_bc, f, mesh, T, N_k, mesh_e = None):

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
    # assemble mass matrix and stiffness matrix    
    a_M = u*v*dx
    a_A = inner(nabla_grad(u), nabla_grad(v))*dx
    M = assemble(a_M)
    A = assemble(a_A)
    B = M + k*A
    
    # Compute solution
    u = Function(V)   # the unknown at a new time level
    t = 0.0
    u_e.t = t

    if mesh_e == None:
        mesh_e = mesh
    err = errornorm(u_e, u_1,mesh=mesh_e)
    
    for i in xrange(N_k):
        t = i*k
        f.t = t # update time for heat source
        f_tk = interpolate(f, V)
        F_tk = f_tk.vector()
        b = M*u_1.vector() + k*M*F_tk # compute r.h.s
        # apply boundary conditions to B and b
        bc.apply(B, b)
        solve(B, u.vector(), b) # compute new solution
        # feed new solution into b
        u_1.assign(u)
    
    # compute error at final time
    u_e.t = t
    err = np.maximum(err, errornorm(u_e, u_1, mesh=mesh_e))

    return err

def compute_err(nx, ny, N_k, nx_max=None):
    if nx_max == None:
        nx_max = max(nx,ny)
    # get parameter values
    lam, T = parameters()
    # get initial condition, boundary condition, heat source, exact sol
    u_ini = exact_sol(lam)
    u_bc = boundaryCond()
    f = heatSource(lam)
    u_e = exact_sol(lam)
    # generate mesh 
    mesh = UnitSquareMesh(nx, ny)
    mesh_e = UnitSquareMesh(nx_max,nx_max)
    
    # numerical solution of Heat Eq with animation
    return L2ErrHeatEq(u_e, u_ini, u_bc, f, mesh, T, N_k, mesh_e)

def print_EOC(h,E,letter = 'h'):
    # Convergence rates
    from math import log as ln  # log is a dolfin name too
    for i in range(1, len(E)):
        r = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
        print letter + '=%8.2E E=%8.2E EOC=%.2f' % (h[i], E[i], r)

def main():
    # first experiment: spatial order of convergence
    N_k = 200 # fixed temporal step size
    E1 = [] # list of erros
    h = []  # element sizes
    print "\nFEM discretization of the heat equation" 
    print "Experimental order of convergence of spatial discretization"
    print "Number of time steps: N_k =", N_k
    for nx in np.ceil(4*np.sqrt(2**np.arange(5))):
        nx = int(nx)
        # compute errors
        E1.append(compute_err(nx, nx, N_k, 256))
        h.append(1./nx)

    print_EOC(h,E1)
    
    # second experiment: temporal order of convergence
    #nx = 800 # fixed number of elements per axis
    #E2 = []
    #k = []
    #print "\nExperimental order of convergence of temporal discretization"
    #print "element size: h =", 1./nx
    #for N_k in 4*2**np.arange(5):
        # compute errors
    #    E2.append(compute_err(nx, nx, N_k))
    #    k.append(1./N_k)

    #print_EOC(k,E2,letter='k')
    
    # third experiment: spatio-temporal order of convergence
    E3 = []
    k = []
    print "\nExperimental order of convergence of full discretization"
    for N_k in 4*4**np.arange(5):
        # compute errors
        nx = 2*int(np.ceil(np.sqrt(N_k)))
        E3.append(compute_err(nx, nx, N_k, 512))
        k.append(1./N_k)

    print_EOC(k,E3,letter='k')



if __name__ == "__main__":
    main()
