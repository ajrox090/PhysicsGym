import math

from phi.torch.flow import *
from src.env.phiflow.heat import Heat
from src.util.burgers_util import simpleGaussianClash

N = 128
num_envs = 1
domain_dict = dict(x=N, y=N, bounds=Box(x=10, y=10), extrapolation=extrapolation.BOUNDARY)

diffusivity = 0.01
dt = 0.1

temperature = CenteredGrid(Sphere(x=5, y=5, radius=1.5), **domain_dict)
physics = Heat(diffusivity=diffusivity)
temperature -= dt * CenteredGrid(Box(x=None, y=(1, 2.5)), **domain_dict)
for _ in view(play=False, framerate=30, namespace=globals()).range():
    temperature = physics.step(temperature, dt=dt)
