""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""
from phi.flow import *

DOMAIN = dict(x=64, extrapolation=extrapolation.PERIODIC)
DT = 1.0
temperature = CenteredGrid(0, **DOMAIN)

for _ in view(temperature, framerate=2, namespace=globals()).range():
    temperature -= DT * CenteredGrid(Box(x=10), **DOMAIN)
    temperature += DT * CenteredGrid(Sphere(x=5, radius=4), **DOMAIN)
    temperature = diffuse.explicit(temperature, 0.5, DT, substeps=4)
