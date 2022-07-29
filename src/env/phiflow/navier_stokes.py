from phi.physics._effect import effect_applied
from phi.physics._physics import StateDependency, Physics
from phi.flow import *  # minimal dependencies
# from phi.torch.flow import *


class NavierStokes(Physics):

    def __init__(self):
        Physics.__init__(self, dependencies=[
            StateDependency('pressure', 'pressure'),
            StateDependency('boundary_mask', 'boundary_mask'),
            StateDependency('obstacles', 'obstacle'),
            StateDependency('velocity_effects', 'velocity_effect', blocking=True),
        ])

    def step(self, v, dt=1.0, boundary_mask=None,
             pressure=None, obstacles=(), velocity_effects=()):
        v = advect.semi_lagrangian(v, v, dt=dt)  # + dt * buoyancy_force
        if boundary_mask is not None:
            v = v * (1 - boundary_mask) + boundary_mask * (2.0, 0)
        for effect in velocity_effects:
            v = effect_applied(effect, v, dt)
        v, pressure = fluid.make_incompressible(v, obstacles, Solve('auto', 1e-5, 0, x0=pressure))
        vorticity = field.curl(v)
        return v, pressure, vorticity

# For Example:
# SPEED = vis.control(2.)
# CYLINDER1 = Obstacle(geom.infinite_cylinder(x=15, y=32, radius=5, inf_dim=None))
# CYLINDER2 = Obstacle(geom.infinite_cylinder(x=15, y=64, radius=5, inf_dim=None))
#
# velocity = StaggeredGrid((SPEED, 0), extrapolation.BOUNDARY, x=128, y=128, bounds=Box(x=128, y=100))
# BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None),
#           velocity.extrapolation, velocity.bounds, velocity.resolution)
# pressure = None
#
# ns = NavierStokes()
# ns.step(velocity=velocity, pressure=pressure)
