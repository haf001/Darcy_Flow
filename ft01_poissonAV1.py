from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Create mesh and define function space
E = []
DOF = []

for m in range (5, 400, 30):
    mesh = UnitSquareMesh(m, m)
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Computing Error in L2 norm for Pressure
    E1 = errornorm(u_D, u, 'L2')
    print('E1 =', E1)
    E.append(E1)
    DOF.append(len(V.dofmap().dofs()))

    # Save solution in VTK format
    #file = File('poissonsolution.pvd')
    #file << u

    # Plot solution
    #import matplotlib.pyplot as plt
    #plot(u)
    #plt.show()

print(E)
print(DOF)
Ea = np.array(E)
DOFa = np.array(DOF)
LogEa = np.log(Ea)
LogDOFa = np.log(DOFa)
print(LogEa)
print(LogDOFa)
x = np.log(DOF)
y = np.log(E)
plt.plot(x,y)
plt.show()
