from phi.physics import diffuse, advect
from phi.physics._effect import effect_applied
from phi.physics._physics import StateDependency, Physics


class Burgers(Physics):

    def __init__(self, default_viscosity=0.1, diffusion_substeps=1):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        self.default_viscosity = default_viscosity
        self.diffusion_substeps = diffusion_substeps
        self.trajectory = []

    def step(self, v, dt=1.0, effects=()):
        trajectory = Burgers.step_velocity(v, self.default_viscosity, dt, effects, self.diffusion_substeps)
        self.trajectory.append(v.vector['x'])

        return trajectory

    @staticmethod
    def step_velocity(v, viscosity, dt, effects, diffusion_substeps):
        v = diffuse.explicit(field=v, diffusivity=dt * viscosity,
                             dt=dt, substeps=diffusion_substeps)
        # v = diffuse.fourier(field=v, diffusivity=viscosity,
        #                     dt=dt)
        v = advect.semi_lagrangian(v, v, dt)
        # v = advect.advect(v, v, dt)

        for effect in effects:
            v = effect_applied(effect, v, dt)
        return v
