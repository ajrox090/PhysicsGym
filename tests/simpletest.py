from phi.flow import *
from functools import partial
import numpy as np
from phi.physics.advect import rk4

N = 128
domain = dict(x=N, y=N, extrapolation=extrapolation.PERIODIC, bounds=Box[-1: 1, -1:1])
domain_pc = dict(extrapolation=extrapolation.PERIODIC, bounds=Box[-1: 1, -1:1])
diffusivity = 0.0005
dt = 0.01
t = 0


def f(x):
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    return u0


u = CenteredGrid(f, **domain)
pu = PointCloud(u.points, **domain_pc)
while t < 1:
    pu = advect.advect(pu, velocity=u,  dt=dt, integrator=rk4)
    t += dt

vis.plot(pu)
vis.show()
# u = diffuse.fourier(u, diffusivity, dt=dt)
# vis.plot(u)
# vis.show()
