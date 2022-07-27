import sys
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from phi.flow import *
from phi.math import pi
import numpy as np

from src.env.phiflow.burgers import Burgers
from src.util.burgers_util import waterfall

N = 32
domain = dict(x=N, extrapolation=extrapolation.PERIODIC)  # , bounds=Box[-1: 1, -1:1])
# domain_pc = dict(extrapolation=extrapolation.PERIODIC, bounds=Box[-1: 1, -1:1])
viscosity = 0.0005
dt = 0.01
t = 0


def myplot(x, i):
    a = x.data._native.reshape(-1)
    plt.plot(a)
    plt.savefig('plots/t{}'.format(i))
    plt.xlim([0, N])
    plt.ylim([0, 1.1])
    plt.show()


def _f1(x):
    dx = 2 * pi / N
    aax = np.arange(-pi, pi, dx).reshape(-1, 1)
    u0 = np.exp(-10 * np.sin(aax / 2) ** 2)
    result = deepcopy(x)
    result._native = deepcopy(u0)
    return result


a = CenteredGrid(_f1, **domain)
physics = Burgers(default_viscosity=viscosity)
if not os.path.exists("plots/"):
    os.makedirs("plots/")

x = a.data._native.reshape(-1)  # initial state
V = []
t = [i for i in range(1, x.size + 1)]
for i in range(10 * N):
    if i % N == 0:
        # V.append(a.data._native.reshape(-1))
        myplot(a, i)
    a = physics.step(a, dt=dt, effects=())

# ax = waterfall(x, t, V, figsize=(8, 8))
# ax.grid(False)
# ax.axis(False)
# ax.view_init(62, -69)
# plt.show()
sys.exit()


def f(x):
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    return u0


u = CenteredGrid(f, **domain)
pu = PointCloud(u.points, **domain_pc)
while t < 1:
    pu = advect.advect(pu, velocity=u, dt=dt, integrator=rk4)
    t += dt

vis.plot(pu)
vis.show()
# u = diffuse.fourier(u, diffusivity, dt=dt)
# vis.plot(u)
# vis.show()
