import time
from fenics import *
from dolfin import dX
import matplotlib.pyplot as plt


def heat_steady_01():
    #  MESH:
    sw = Point(0.0, 0.0)
    ne = Point(5.0, 1.0)
    mesh = RectangleMesh(sw, ne, 50, 10)

    #  FUNCTION SPACE:
    V = FunctionSpace(mesh, "Lagrange", 1)

    #  BOUNDARY CONDITIONS:
    y_top = 1.0
    u_top = 10.0

    def on_top(x, on_boundary):
        return y_top - DOLFIN_EPS <= x[1]

    bc_top = DirichletBC(V, u_top, on_top)

    u_side = 100.0

    def on_side(x, on_boundary):
        return on_boundary and x[1] < y_top - DOLFIN_EPS

    bc_side = DirichletBC(V, u_side, on_side)

    bc = [bc_top, bc_side]

    #  TRIAL and TEST FUNCTIONS:
    u = TrialFunction(V)
    v = TestFunction(V)

    #  BILINEAR and LINEAR FORMS:
    k = Constant(1.0)
    Auv = k * inner(grad(u), grad(v)) * dX

    f = Constant(0.0)
    Lv = f * v * dX

    #  SOLVE: the variational problem a(u,v)=l(v) with boundary conditions.
    w = Function(V)
    solve(Auv == Lv, w, bc)
    #  Plot the solution W.
    plot(w, title='heat_steady_01')
    filename = '../heat_steady_01.png'
    plt.savefig(filename)
    print("Saving graphics in file '%s'" % (filename))
    plt.close()

    #  Terminate.
    return


def heat_steady_01_test():

    print(time.ctime(time.time()))
    #  Report level = only warnings or higher.
    level = 30
    set_log_level(level)

    print('')
    print('heat_steady_01_test:')
    print('  FENICS/Python version')
    print('  2D steady heat equation in a rectangle.')

    heat_steady_01()

    #  Terminate.
    print('')
    print('heat_steady_01_test:')
    print('  Normal end of execution.')
    print('')
    print(time.ctime(time.time()))
    return


if __name__ == '__main__':
    heat_steady_01_test()
