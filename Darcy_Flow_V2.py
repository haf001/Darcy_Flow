from fenics import *
from dolfin import *
import numpy as np
from ufl import nabla_div
import sympy as sym
import matplotlib.pyplot as plt

def solver(f, p_e, mesh, degree=1):
    """
    Solving the Darcy flow equation on a unit square media with pressure boundary conditions.
    """

    # Creating mesh and defining function space
    V = FunctionSpace(mesh, 'P', degree)

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

    # If p = p_e on the boundary, then:-
    #def boundary(x, on_boundary):
        #return on_boundary

    #bc = DirichletBC(V, p_e, boundary)

    # Defining variational problem
    p = TrialFunction(V)
    v = TestFunction(V)
    d = 2
    I = Identity(d)
    x, y = SpatialCoordinate(mesh)
    M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
    K = M*I
    a = dot(K*grad(p), grad(v))*dx
    L = inner(f, v)*dx

    # Computing Numerical Pressure
    p = Function(V)
    solve(a == L, p, bcs)

    return p

def run_solver():
    "Run solver to compute and post-process solution"

    mesh = UnitSquareMesh(40, 40)

    # Setting up problem parameters and calling solver
    x, y = sym.symbols('x[0], x[1]')
    d = 2
    I = Identity(d)
    x, y = SpatialCoordinate(mesh)
    M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
    K = M*I
    p_e = Expression('1 - x[0]*x[0]', degree=2)
    #p_e = Expression('sin(x[0])*sin(x[1])', degree=2)
    f1 = as_vector((-2*x, Constant(0)))
    #f1 = as_vector((cos(x)*sin(y), sin(x)*cos(y)))
    f = nabla_div(dot(-K, f1))
    p = solver(f, p_e, mesh, 1)
    u_bar = -K*grad(p)
    u_bar1 = project(u_bar, VectorFunctionSpace(mesh, 'P', degree=1))

    # Saving and Plotting Numerical solutions for visualization
    xdmf = XDMFFile('Numerical_Pressure_Gradient_V2.xdmf')
    xdmf.write(p)
    xdmf.close()

    xdmf = XDMFFile('Velocity_Profile_V2.xdmf')
    xdmf.write(u_bar1)
    xdmf.close()

def test_solver():
    "Test solver by reproducing p = 1 - x^2"

    # Set up parameters for testing
    p_e = Expression('1 - x[0]*x[0]', degree=2)
    #p_e = Expression('sin(x[0])*sin(x[1])', degree=2)

    # Iterating over mesh sizes and DOF
    E = []
    DOF = []
    for m in range (10, 300, 10):
        mesh = UnitSquareMesh(m, m)
        V = FunctionSpace(mesh, 'P', 1)
        p_e_f = interpolate(p_e, FunctionSpace(mesh, 'P', 2))
        x, y = sym.symbols('x[0], x[1]')
        d = 2
        I = Identity(d)
        x, y = SpatialCoordinate(mesh)
        M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
        K = M*I
        f1 = as_vector((-2*x, 0))
        #f1 = as_vector((cos(x)*sin(y), sin(x)*cos(y)))
        f = nabla_div(dot(-K, f1))

        # Compute solution
        p = solver(f, p_e, mesh, degree=1)

        # Computing Error in L2 norm for Pressure
        E1 = errornorm(p_e_f, p, 'L2')
        print('E1 =', E1)
        E.append(E1)
        DOF.append(len(V.dofmap().dofs()))

    print(E)
    print(DOF)
    Ea = np.array(E)
    DOFa = np.array(DOF)
    LogEa = np.log(Ea)
    LogDOFa = np.log(DOFa)

    return (E, DOF)

if __name__ == '__main__':
    run_solver()

    E, DOF = test_solver()

    # Log plot of L2 Error against DOF
    x = np.log(DOF)
    y = np.log(E)
    plt.plot(x,y)
    plt.title('Log L2 Error vs. Log DOF')
    plt.xlabel("Log DOF")
    plt.ylabel("Log L2 Error")
    plt.show()

    # Semilog plot of L2 Error against DOF
    plt.semilogy(DOF, E)
    plt.title('Semilog L2 Error vs DOF')
    plt.xlabel("DOF")
    plt.ylabel("L2 Error")
    plt.show()
