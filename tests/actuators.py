from phi.flow import *
import phi.math as pmath

v = CenteredGrid(0, x=10, y=10, bounds=Box(x=40, y=40), extrapolation=0)


def f(x):
    """ Actuation function used in Model predictive control for PDE thesis, for Catalytic rod experiment."""
    return pmath.sqrt(2 / pmath.pi) * pmath.sin(x)


# some random
def f2(x):
    return pmath.sin(x ** 2) * pmath.cos(x ** 3)


actuator = CenteredGrid(f2, x=50, y=50, bounds=Box(x=100, y=100), extrapolation=extrapolation.PERIODIC)

vis.show(actuator)
