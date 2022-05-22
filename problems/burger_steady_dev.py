from fenics import *
import matplotlib.pyplot as plt
from problems.BaseSteady import BaseSteady

level = 30
set_log_level(level)


class BurgerSteady(BaseSteady):

    def __init__(self, V=None, bc=None, nu=0.1, savfig=True, e_num=16):
        super().__init__()
        print('')
        print('burgers_steady_viscous:')
        self.V = V
        self.bc = bc
        self.nu = nu
        self.e_num = e_num
        self.savfig = savfig

    def define(self):

        print('  Number of elements is %d' % self.e_num)
        print('  Viscosity set to %g' % self.nu)

        x_left = -1.0
        x_right = +1.0
        mesh = IntervalMesh(self.e_num, x_left, x_right)
        V = FunctionSpace(mesh, "CG", 1)
        self.V = V

        def on_left(x, on_boundary):
            return x[0] <= x_left + DOLFIN_EPS

        def on_right(x, on_boundary):
            return x_right - DOLFIN_EPS <= x[0]

        u_left = -1.0
        u_right = +1.0
        bc_left = DirichletBC(V, u_left, on_left)
        bc_right = DirichletBC(V, u_right, on_right)
        bc = [bc_left, bc_right]

        self.bc = bc

    def solve(self):

        u = Function(self.V)
        v = TestFunction(self.V)
        F = \
            ( \
                        self.nu * inner(grad(u), grad(v)) \
                        + inner(u * u.dx(0), v) \
                ) * dX

        J = derivative(F, u)

        solve(F == 0, u, self.bc, J=J)

        if self.savfig:
            plot(u, title='burgers steady viscous equation')
            filename = 'burgers_steady_viscous.png'
            plt.savefig(filename)
            print('Graphics saved as "%s"' % (filename))
            plt.close()
