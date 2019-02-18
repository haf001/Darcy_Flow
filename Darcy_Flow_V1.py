from fenics import *
import numpy as np
from ufl import nabla_div, max_value
import sympy as sym

x, y = sym.symbols('x[0], x[1]')

# Creating mesh and defining function space
E = []
DOF = []
mesh = UnitSquareMesh(60, 60)
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
x, y = SpatialCoordinate(mesh)
M = max_value(Constant(0.10), exp(-(10*y-1.0*sin(10*x)-5.0)**2))
K = M*I
a = dot(K*grad(p), grad(v))*dx
f1 = as_vector((-2*x, 0))
f = nabla_div(dot(-K, f1))
L = inner(f, v)*dx

# Computing solutions
p = Function(V)
solve(a == L, p, bcs)
u_bar = -K*grad(p)

# Projecting the Velocity profile
W = VectorFunctionSpace(mesh, 'P', 1)
u_bar1 = project(u_bar, W)

# Evaluating difference between Numerical pressure, p and Exact pressure, p_e
p_e_f = interpolate(p_e, V)
diff_p = Function(V)
diff_p.vector()[:] = p.vector() - p_e_f.vector()

# Computing Error in L2 norm for Pressure
E1 = errornorm(p_e_f, p, 'L2')
print('E1 =', E1)
E.append(E1)
DOF.append(len(V.dofmap().dofs()))

# Plotting Pressure Gradient, Difference_in_Numeric(p)_&_Exact_Pressure(p_e)_and Velocity_Profile
xdmf = XDMFFile('Numerical_Pressure_Gradient.xdmf')
xdmf.write(p)
xdmf.close()

xdmf = XDMFFile('Difference_in_Pressures.xdmf')
xdmf.write(diff_p)
xdmf.close()

xdmf = XDMFFile('Analytical_Pressure_Gradient.xdmf')
xdmf.write(p_e_f)
xdmf.close()

xdmf = XDMFFile('Velocity_Profile.xdmf')
xdmf.write(u_bar1)
xdmf.close()

print(E)
print(DOF)
Ea = np.array(E)
DOFa = np.array(DOF)
LogEa = np.log(Ea)
LogDOFa = np.log(DOFa)
print(LogEa)
print(LogDOFa)
print(type(p))
print(type(p_e_f))
