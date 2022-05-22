import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from BaseSteady import BaseSteady

level = 30
set_log_level(level)


class Poisson(BaseSteady):

    def __init__(self, savfig=True):
        super(Poisson, self).__init__()
        self.V = None
        self.bc = None
        self.u_D = None
        self.mesh = None
        self.savfig = savfig

    def define(self):
        mesh = UnitSquareMesh(8, 8)
        self.mesh = mesh
        V = FunctionSpace(mesh, 'P', 1)
        self.V = V
        u_D = Expression('1 + x[0]*x[0] + 6*x[1]*x[1]', degree=4)

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u_D, boundary)
        self.bc = bc
        self.u_D = u_D

    def solve(self):

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        f = Constant(-14.0)
        a = dot(grad(u), grad(v)) * dX
        L = f * v * dX

        u = Function(self.V)
        solve(a == L, u, self.bc)

        if self.savfig:
            plot(self.mesh, title='Mesh for Poisson equation')
            filename = 'poisson_mesh.png'
            plt.savefig(filename)
            print('  Graphics saved as "%s"' % (filename))
            plt.close()

            plot(u, mode='contour', title='Solution for Poisson equation')
            filename = 'poisson_solution.png'
            plt.savefig(filename)
            print('  Graphics saved as "%s"' % (filename))
            plt.close()

        error_L2 = errornorm(self.u_D, u, 'L2')
        print('  error_L2  =', error_L2)

        vertex_values_u_D = self.u_D.compute_vertex_values(self.mesh)
        vertex_values_u = u.compute_vertex_values(self.mesh)
        error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
        print('  error_max =', error_max)

