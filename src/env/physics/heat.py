from phi.physics import diffuse
from phi.physics._effect import effect_applied
from phi.physics._physics import Physics, StateDependency


class Heat(Physics):

    def __init__(self, diffusivity=0.1):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        self.diffusivity = diffusivity

    def step(self, t, dt=1.0, effects=()):
        return Heat.step_velocity(t, self.diffusivity, dt, effects)

    @staticmethod
    def step_velocity(t, diffusivity, dt, effects):
        t = diffuse.explicit(field=t, diffusivity=dt * diffusivity,
                             dt=dt)
        for effect in effects:
            t = effect_applied(effect, t, dt)
        return t
