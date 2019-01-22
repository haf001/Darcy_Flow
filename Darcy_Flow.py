from fenics import *

# Creating mesh and defining function space
mesh = UnitSquareMesh(24, 24)
V = FunctionSpace(mesh, "P", 1)

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
a = dot(K*grad(p), grad(v))*dx
f = Constant(0.0)
L = f*v*dx

# Computing solutions
p = Function(V)
solve(a == L, p, bcs)
u_bar = -K*grad(p)

# Projecting the Velocity vector
W = VectorFunctionSpace(mesh, "P", 1)
u_bar1 = project(u_bar, W)

# Saving Solution in VTK format
file = File('Pressure_Gradient.pvd')
file << p
file = File('Velocity_Profile.pvd')
file << u_bar1
