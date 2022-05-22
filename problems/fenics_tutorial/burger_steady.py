import time
from fenics import *
import matplotlib.pyplot as plt


def burgers_steady_viscous(e_num, nu):
    print('')
    print('burgers_steady_viscous:')
    print('  FENICS/Python version')
    print('  Number of elements is %d' % (e_num))
    print('  Viscosity set to %g' % (nu))

    x_left = -1.0
    x_right = +1.0
    mesh = IntervalMesh(e_num, x_left, x_right)
    V = FunctionSpace(mesh, "CG", 1)
    u_left = -1.0
    u_right = +1.0

    def on_left(x, on_boundary):
        return (x[0] <= x_left + DOLFIN_EPS)

    def on_right(x, on_boundary):
        return (x_right - DOLFIN_EPS <= x[0])

    bc_left = DirichletBC(V, u_left, on_left)
    bc_right = DirichletBC(V, u_right, on_right)
    bc = [bc_left, bc_right]

    u = Function(V)
    v = TestFunction(V)
    F = \
        ( \
                    nu * inner(grad(u), grad(v)) \
                    + inner(u * u.dx(0), v) \
            ) * dX

    J = derivative(F, u)

    solve(F == 0, u, bc, J=J)

    plot(u, title='burgers steady viscous equation')
    filename = '../burgers_steady_viscous.png'
    plt.savefig(filename)
    print('Graphics saved as "%s"' % (filename))
    plt.close()

    return


def burgers_steady_viscous_test():
    print(time.ctime(time.time()))
    level = 30
    set_log_level(level)

    print('')
    print('burgers_steady_viscous_test:')
    print('  FENICES/Python version')
    print('  Solve the steady 1d Burgers equation.')

    e_num = 16
    nu = 0.1
    burgers_steady_viscous(e_num, nu)

    print("")
    print("burgers_steady_viscous_test:")
    print("  Normal end of execution.")
    print('')
    print(time.ctime(time.time()))
    return


if __name__ == '__main__':
    burgers_steady_viscous_test()
