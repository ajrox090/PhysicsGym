""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""
import random

from phi.flow import *
from phi import math


def GaussianForce(x):
    return 3.4 * math.exp(-0.5 * (x - 0.34) ** 2 / 0.34 ** 2)


def gaussian(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2)
    right = math.exp(-0.5 * (x - 0.34) ** 2)
    result = left + right
    return result


DOMAIN = dict(x=64, y=64, bounds=Box(x=2 * math.pi, y=2 * math.pi), extrapolation=extrapolation.PERIODIC)
dt = 0.1

sim_velocity = StaggeredGrid(GaussianForce, **DOMAIN)  # also works with CenteredGrid

viewer = view(sim_velocity, namespace=globals(), play=False)
for _ in viewer.range():
    sim_velocity = diffuse.explicit(sim_velocity, diffusivity=0.01, dt=dt)
    sim_velocity = advect.semi_lagrangian(sim_velocity, sim_velocity, dt)
