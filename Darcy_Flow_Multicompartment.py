from fenics import *
import numpy as np
from ufl import nabla_div
import sympy as sym
import matplotlib.pyplot as plt

def solver(f, b_12_val, b_23_val, mesh, degree=2):
    """
    Solving the Reduced Darcy multi-compartment model for 3 equal sized
    porous media with domain of Omega = [0,1] x [0,1] using pressure boundary
    conditions and specified intercompartment coupling coefficients listed
    Beta_11 = Beta_22 = Bta_33 = Beta_21 = Beta_32 = Beta_13 = Beta_31 = 0.00,
    Beta_12 = Constant value 0.02, Beta_23 = Constant value 0.03
    """

    # Creating mesh and defining function space
    Velm = FiniteElement('P', mesh.ufl_cell(), degree)
    Welm = MixedElement([Velm, Velm, Velm])
    W = FunctionSpace(mesh, Welm)

    # Defining Boundary Conditions
    p_L = Constant(1.0)

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0)

    bc_L = DirichletBC(W.sub(0), p_L, boundary_L)

    p_R = Constant(0.0)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1)

    bc_R = DirichletBC(W.sub(2), p_R, boundary_R)

    # Collecting boundary conditions
    bcs = [bc_L, bc_R]

    # Defining variational problem
    (p_1, p_2, p_3) = TrialFunctions(W)
    (v_1, v_2, v_3) = TestFunctions(W)

    f_1, f_2, f_3 = f
    d = 2
    I = Identity(d)
    M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
    K_1 = M*I
    K_2 = M*I
    K_3 = M*I
    b_12 = Constant(b_12_val)
    b_23 = Constant(b_23_val)
    a = dot(K_1*grad(p_1), grad(v_1))*dx + dot(K_2*grad(p_2), grad(v_2))*dx + dot(K_3*grad(p_3), grad(v_3))*dx + (b_12*v_1*(p_1 - p_2))*dx + (b_23*v_2*(p_2 - p_3))*dx
    L = inner(f_1, v_1)*dx + inner(f_2, v_2)*dx + inner(f_3, v_3)*dx

    # Computing Numerical Pressure
    p = Function(W)
    solve(a == L, p, bcs)

    return p

def run_solver():
    "Run solver to compute and post-process solution"

    mesh = UnitSquareMesh(40, 40)

    # Setting up problem parameters and calling solver
    d = 2
    I = Identity(d)
    M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
    K_1 = M*I
    K_2 = M*I
    K_3 = M*I
    b_12_val = 0.02
    b_23_val = 0.03
    b_12 = Constant(b_12_val)
    b_23 = Constant(b_23_val)
    p_1 = Expression('1 - x[0]*x[0]', degree=2)
    p_2 = Expression('1 - x[0]*x[0]', degree=2)
    p_3 = Expression('1 - x[0]*x[0]', degree=2)

    f_1 = Expression(('-2*x[0]', '0.0'), degree=1, domain=mesh)
    f_2 = Expression(('-2*x[0]', '0.0'), degree=1, domain=mesh)
    f_3 = Expression(('-2*x[0]', '0.0'), degree=1, domain=mesh)
    #f_2 = Expression(('cos(x[0])*sin(x[1])', 'sin(x[0])*cos(x[1])'), degree=2, domain=mesh)
    #f_3 = Expression(('cos(x[0])*sin(x[1])', 'sin(x[0])*cos(x[1])'), degree=2, domain=mesh)

    f1 = -nabla_div(dot(K_1, f_1)) + b_12*(p_1 - p_2)
    f2 = -nabla_div(dot(K_2, f_2)) + b_23*(p_2 - p_3)
    f3 = -nabla_div(dot(K_3, f_3))

    p = solver((f1, f2, f3), b_12_val, b_23_val, mesh, 2)
    # p here is a coefficient vector of mixed function space. To get components

    p_1, p_2, p_3 = p.split(deepcopy=True)
    u_bar1 = -K_1*grad(p_1)
    u_bar2 = -K_2*grad(p_2)
    u_bar3 = -K_3*grad(p_3)
    u_bar1_1 = project(u_bar1, VectorFunctionSpace(mesh, 'P', degree=2))
    u_bar2_2 = project(u_bar2, VectorFunctionSpace(mesh, 'P', degree=2))
    u_bar3_3 = project(u_bar3, VectorFunctionSpace(mesh, 'P', degree=2))

    # Saving and Plotting Numerical solutions for visualization
    xdmf = XDMFFile('Numerical_Pressure_Gradient_p_1.xdmf')
    xdmf.write(p_1)
    xdmf.close()

    xdmf = XDMFFile('Numerical_Pressure_Gradient_p_2.xdmf')
    xdmf.write(p_2)
    xdmf.close()

    xdmf = XDMFFile('Numerical_Pressure_Gradient_p_3.xdmf')
    xdmf.write(p_3)
    xdmf.close()

    xdmf = XDMFFile('Velocity_Profile_1.xdmf')
    xdmf.write(u_bar1_1)
    xdmf.close()

    xdmf = XDMFFile('Velocity_Profile_2.xdmf')
    xdmf.write(u_bar2_2)
    xdmf.close()

    xdmf = XDMFFile('Velocity_Profile_3.xdmf')
    xdmf.write(u_bar3_3)
    xdmf.close()

if __name__ == '__main__':
    run_solver()
