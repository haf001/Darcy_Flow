from fenics import *
import numpy as np

# Creating mesh and defining function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Defining Exact Solution for Pressure Distribution
p_e = Expression('1 - x[0]', degree=1)

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

p_e_f = interpolate(p_e, V)
diff_p = Function(V)
diff_p.vector()[:] = p.vector() - p_e_f.vector()

# Saving Solution for Pressure in VTK format
file = File('Pressure_Gradient.pvd')
file << p
File('diff_p.pvd') << diff_p

def compute_errors(p_e, p):

    # Computing Error in L2 norm for Pressure
    E1 = errornorm(p_e, p, 'L2')
    #print('E1 =', E1)

    # Computing Error in H1 seminorm for Pressure
    E2 = errornorm(p_e, p, 'H10')
    #print('E2 =', E2)

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'L2 norm': E1,
          'H10 seminorm': E2}

    return errors

def compute_convergence_rates(p_e, f, p, K,
                          max_degree=3, num_levels=5):
    #"Compute convergences rates for various error norms"

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level][error_type]

    # Iterating over degrees and mesh refinement levels
    degrees = range(1, max_degree + 1)
    for degree in degrees:
        n = 16  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            h[degree].append(1.0 / n)
            errors = compute_errors(p_e, p)
            E[degree].append(errors)
            print('2 x (%d x %d) P%d mesh, %d unknowns, E1 = %g, E2 = %g' %
            (n, n, degree, p.function_space().dim(), errors['L2 norm'], errors['H10 seminorm']))
            n *= 2

    # Compute convergence rates
    from math import log as ln
    etypes = list(E[1][0].keys())
    rates = {}
    for degree in degrees:
        rates[degree] = {}
        for error_type in sorted(etypes):
            rates[degree][error_type] = []
            for i in range(1, num_levels):
                Ei = E[degree][i][error_type]
                Eim1 = E[degree][i - 1][error_type]
                r = ln(Ei / Eim1) / ln(h[degree][i] / h[degree][i - 1])
                rates[degree][error_type].append(round(r, 2))
    return etypes, degrees, rates

# Compute and print convergence rates
etypes, degrees, rates = compute_convergence_rates(p_e, f, p, K)
for error_type in etypes:
    print('\n' + error_type)
    for degree in degrees:
        print('P%d: %s' % (degree, str(rates[degree][error_type])[1:-1]))

# Computing maximum error at vertices for Pressure
vertex_values_p = p.compute_vertex_values(mesh)
vertex_values_p_e = p_e.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_p_e - vertex_values_p))
print('error_max =', error_max)

#def flux(p, K):
    #"Returning -K*grad(p) to be projected into same space as Pressure p"
    #V = p.function_space()
    #mesh = V.mesh()
    #degree = V.ufl_element().degree()
    #W = VectorFunctionSpace(mesh, 'P', degree)
    #flux_u = project(-K*grad(p), W)
    #return flux_u
    #file = File('Velocity_Profile.pvd')
    #file << flux_u

#flux_u = flux(p, K)
#plot(flux_u, title='flux field')
#plt.show()
#flux_u_x, flux_u_y = flux_u.split(deepcopy=True)
#plot(flux_u_x, title='x-component of flux (-K*grad(p))')
#plt.show()
#plot(flux_u_y, title='y-component of flux (-K*grad(p))')
#plt.show()

# Exact flux expressions
#p_e = lambda x, y: 1 - x
#flux_x_exact = lambda x, y: K
#flux_y_exact = lambda x, y: O

# Computing the error in flux
#coor = p.function_space().mesh().coordinates()
#for i, value in enumerate(flux_u_x.compute_vertex_values()):
    #print('vertex %d, x = %s, -p*u_x = %g, error = %g' %
          #(i, tuple(coor[i]), value, flux_x_exact(*coor[i]) - value))
#for i, value in enumerate(flux_u_y.compute_vertex_values()):
    #print('vertex %d, x = %s, -p*u_y = %g, error = %g' %
          #(i, tuple(coor[i]), value, flux_y_exact(*coor[i]) - value))
