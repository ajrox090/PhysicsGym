from phi.physics import diffuse, advect, fluid
from phi.physics._effect import effect_applied
from phi.physics._physics import StateDependency, Physics


class NavierStokes(Physics):

    def __init__(self, viscosity=0.1, buoyancy_factor=1.0):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        self.viscosity = viscosity
        self.buoyancy_factor = buoyancy_factor

    def step(self, v, dt=1.0, effects=()):
        return NavierStokes.step_forward(velocity=v, viscosity=self.viscosity,
                                         dt=dt, effects=effects)

    @staticmethod
    def step_forward(velocity, viscosity, dt=1.0, effects=()):
        # smoke = advect.semi_lagrangian(smoke, velocity, dt) + INFLOW
        # smoke = advect.semi_lagrangian(smoke, velocity, dt)
        # buoyancy_force = (smoke * (0, self.buoyancy_factor)).at(velocity)  # resamples smoke to velocity sample points
        velocity = advect.semi_lagrangian(velocity, velocity, dt)   # + dt * buoyancy_force
        velocity = diffuse.explicit(velocity, viscosity, dt)
        velocity, pressure = fluid.make_incompressible(velocity)
        return velocity, pressure
        # return velocity, smoke, pressure
