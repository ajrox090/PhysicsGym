"""Karman Vortex Street
Air flow around a static cylinder.
Vortices start appearing after a couple of hundred steps.
"""
# from phi.flow import *  # minimal dependencies
from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


SPEED = vis.control(2.)
CYLINDER1 = Obstacle(geom.infinite_cylinder(x=15, y=32, radius=5, inf_dim=None))
CYLINDER2 = Obstacle(geom.infinite_cylinder(x=15, y=64, radius=5, inf_dim=None))

velocity = StaggeredGrid((SPEED, 0), extrapolation.BOUNDARY, x=128, y=128, bounds=Box(x=128, y=100))
# vis.show(velocity, title='Initial velocity')
BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None), velocity.extrapolation, velocity.bounds, velocity.resolution)
# vis.show(BOUNDARY_MASK, title='Boundary mask')

pressure = None


@math.jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, p, dt=1.):
    v = advect.semi_lagrangian(v, v, dt)
    # vis.show(v, title='advect')
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    # vis.show(v, title='after boundary')
    return fluid.make_incompressible(v, [CYLINDER1, CYLINDER2], Solve('auto', 1e-5, 0, x0=p))


for _ in view('vorticity,velocity,pressure', namespace=globals()).range():
# for i in range(3):
    velocity, pressure = step(velocity, pressure)
    # vis.show(velocity, title='Made incompressible')
    vorticity = field.curl(velocity)
    # vis.show(vorticity, title='voriticity')
    # print(i)
