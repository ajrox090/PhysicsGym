from fenics import *
import matplotlib.pyplot as plt

from problems.BaseSteady import BaseSteady

level = 30
set_log_level(level)


class HeatSteady(BaseSteady):

    def __init__(self, savfig=True):
        super().__init__()
        self.V = None
        self.savfig = savfig
        self.bc = None
        self.w = None

    def define(self):
        sw = Point(0.0, 0.0)
        ne = Point(5.0, 1.0)
        mesh = RectangleMesh(sw, ne, 50, 10)

        V = FunctionSpace(mesh, "Lagrange", 1)
        self.V = V

        y_top = 1.0
        u_top = 10.0
        u_side = 100.0

        def on_top(x, on_boundary):
            return y_top - DOLFIN_EPS <= x[1]

        def on_side(x, on_boundary):
            return on_boundary and x[1] < y_top - DOLFIN_EPS

        bc_top = DirichletBC(V, u_top, on_top)
        bc_side = DirichletBC(V, u_side, on_side)
        bc = [bc_top, bc_side]
        self.bc = bc


    def solve(self):

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        k = Constant(1.0)
        f = Constant(0.0)
        Auv = k * inner(grad(u), grad(v)) * dX
        Lv = f * v * dX
        w = Function(self.V)
        solve(Auv == Lv, w, self.bc)
        self.w = w

        if self.savfig:
            plot(w, title='heat_steady_01')
            filename = 'heat_steady_01.png'
            plt.savefig(filename)
            print("Saving graphics in file '%s'" % (filename))
            plt.close()
        print("solver finished, terminating.")

