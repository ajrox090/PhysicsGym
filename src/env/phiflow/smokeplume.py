from matplotlib import pyplot as plt
from phi.flow import *
from phi.physics._effect import effect_applied
from phi.physics._physics import Physics, StateDependency


class SmokePlume(Physics):

    def __init__(self, velocity, inflow):
        Physics.__init__(self, dependencies=[
            StateDependency('smoke_prev', 'smoke_prev', blocking=True),
            StateDependency('pressure', 'pressure'),
            StateDependency('velocity_effects', 'velocity_effect', blocking=True),
        ])

        self.velocity = velocity
        self.inflow = inflow
        self.pressure = None

    def step(self, velocity_prev, dt=1.0, smoke_prev=None, velocity_effects=()):
        smoke_next = advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + self.inflow
        buoyancy_force = smoke_next * (0.0, 0.1) @ self.velocity
        velocity_tent = advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + \
                        buoyancy_force * dt
        for effect in velocity_effects:
            velocity_tent = effect_applied(effect, velocity_tent, dt)
        velocity_next, self.pressure = fluid.make_incompressible(velocity_tent,
                                                                 Solve('auto', 1e-5, 0, x0=self.pressure))
        return velocity_next, smoke_next

# Example
# from tqdm import tqdm
# velocity = StaggeredGrid(values=(0.0, 0.0), extrapolation=0, x=64, y=64,
#                          bounds=Box(x=100, y=100))
# smoke = CenteredGrid(values=0.0, extrapolation=extrapolation.BOUNDARY,
#                      x=200, y=200, bounds=Box(x=100, y=100))
# inflow = 0.2 * CenteredGrid(values=SoftGeometryMask(Sphere(x=40, y=9.5, radius=5)),
#                             extrapolation=0.0, bounds=smoke.bounds, resolution=smoke.resolution)
#
# sp = SmokePlume(velocity, inflow)
# plt.style.use("dark_background")
#
# for _ in tqdm(range(150)):
#     velocity, smoke = sp.step(velocity_prev=velocity, smoke_prev=smoke, dt=1.0)
#     smoke_values_extracted = smoke.values.numpy("y,x")
#     plt.imshow(smoke_values_extracted, origin="lower")
#     plt.draw()
#     plt.pause(0.01)
#     plt.clf()
