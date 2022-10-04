import copy

from phi.flow import *

SPEED = 2.0
velocity = StaggeredGrid((SPEED, 0), extrapolation.BOUNDARY, x=128, y=128, bounds=Box(x=128, y=64))
CYLINDER = Obstacle(geom.infinite_cylinder(x=15, y=32, radius=5, inf_dim=None))
BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None), velocity.extrapolation, velocity.bounds, velocity.resolution)


def step(v, p, dt=1.):
    # v = advect.semi_lagrangian(v, v, dt)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    vis.show(v)
    return fluid.make_incompressible(v, [CYLINDER], Solve('auto', 1e-5, 0, x0=p))


pressure = None
# for _ in view("velocity, pressure", gui='dash', framerate=20, namespace=globals()).range():
for i in range(100):
    velocity, pressure = step(velocity, pressure)

