from fenics import *
import numpy as np
from ufl import nabla_div

# Creating mesh and defining function space
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, 'P', 1)

# Defining Exact Solution for Pressure Distribution
p_e = Expression('1 - x[0]*x[0]', degree=2)

# Defining Dirichlet boundary
p_L = Constant(1.0)

def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], 0)

bc_L = DirichletBC(V, p_L, boundary_L)

p_R = Constant(0.0)

def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], 1)

bc_R = DirichletBC(V, p_R, boundary_R)

bcs = [bc_L, bc_R]

# Defining variational problem
p = TrialFunction(V)
v = TestFunction(V)
d = 2
I = Identity(d)
M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10*x[0])-5.0, 2)))', degree=2)
K = M*I
#K = 1
a = dot(K*grad(p), grad(v))*dx
f1 = Expression('2*x[0]', degree=1)
f = nabla_div(K * interpolate(f1, V))
#f = Constant(0.0)
#L = f*v*dx
L = f*v*dx
#L = inner(f, v)*dx

# Computing solutions
p = Function(V)
solve(a == L, p, bcs)

p_e_f = interpolate(p_e, V)
diff_p = Function(V)
diff_p.vector()[:] = p.vector() - p_e_f.vector()

# Saving Solution in VTK format
file = File('Pressure_Gradient.pvd')
file << p
File('diff_p.pvd') << diff_p
