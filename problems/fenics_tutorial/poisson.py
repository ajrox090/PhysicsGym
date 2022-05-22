import time
from fenics import *
import matplotlib.pyplot as plt

level = 30
set_log_level(level)


def poisson():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, 'P', 1)
    u_D = Expression('1 + x[0]*x[0] + 6*x[1]*x[1]', degree=4)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-14.0)
    a = dot(grad(u), grad(v)) * dX
    L = f * v * dX

    u = Function(V)
    solve(a == L, u, bc)

    plot(mesh, title='Mesh for Poisson equation')
    filename = '../poisson_mesh.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.close()

    plot(u, mode='contour', title='Solution for Poisson equation')
    filename = '../poisson_solution.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.close()

    error_L2 = errornorm(u_D, u, 'L2')
    print('  error_L2  =', error_L2)

    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    import numpy as np
    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    print('  error_max =', error_max)

    return


def poisson_test():
    print(time.ctime(time.time()))
    print('')
    print('poisson_test:')
    print('  FENICS/Python version')
    print('  Poisson equation on the unit square.')

    poisson()

    print('')
    print('poisson_test:')
    print('  Normal end of execution.')
    print('')
    print(time.ctime(time.time()))
    return


if (__name__ == '__main__'):
    poisson_test()
