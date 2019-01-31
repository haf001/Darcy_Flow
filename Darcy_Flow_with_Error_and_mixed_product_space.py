

from fenics import *

# Create mesh
mesh = UnitSquareMesh(32, 32)

# Define function spaces and mixed (product) space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)

# Defining Exact Solution for Pressure Distribution
p_e = Expression('1 - x[0]', degree=1)

# Define trial and test functions
(u_bar, p) = TrialFunctions(W)
(tau, v) = TestFunctions(W)
d = 2
I = Identity(d)
M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10*x[0])-5.0, 2)))', degree=2)
K = M*I
f = Constant(0.0)
g = Constant(0.0)

# Defining the variational form
a = (dot(u_bar, tau) + dot(K*grad(p), tau) + dot(u_bar, grad(v)))*dx
L = - f*v*dx - g*v*ds

# Defining the Dirichlet Boundary Conditions
p_L = Constant(1.0)

def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], 0)

bc_L = DirichletBC(W.sub(1), p_L, boundary_L)

p_R = Constant(0.0)

def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], 1)

bc_R = DirichletBC(W.sub(1), p_R, boundary_R)

bcs = [bc_L, bc_R]

# Computing the solution
w = Function(W)
solve(a == L, w, bcs)
(u_bar, p) = w.split()

# Plot u_bar and p
xdmf = XDMFFile('Pressure_Gradient.xdmf')

xdmf.write(p)
xdmf.close()

xdmf = XDMFFile('Velocity_Profile.xdmf')
xdmf.write(u_bar)
xdmf.close()
