import sys

from phi.math import pi
from copy import deepcopy

from matplotlib import pyplot as plt
from phi.flow import *
from src.env.phiflow.burgers import Burgers

N = 32
# domain_dict = dict(x=N, y=N, bounds=Box(x=10, y=10), extrapolation=extrapolation.ZERO)
domain_dict = dict(x=N, extrapolation=extrapolation.PERIODIC)
step_count = 1
# viscosity = 0.01 / (N * np.pi)
viscosity = 0.0005
dt = 0.01
diffusion_substeps = 1


def officialGaussianClash(x):
    batch_size = 32
    leftloc = np.random.uniform(0.2, 0.4)
    leftamp = np.random.uniform(0, 3)
    leftsig = np.random.uniform(0.05, 0.15)
    rightloc = np.random.uniform(0.6, 0.8)
    rightamp = np.random.uniform(-3, 0)
    rightsig = np.random.uniform(0.05, 0.15)
    left = leftamp * math.exp(-0.5 * (x.vector[0] - leftloc) ** 2 / leftsig ** 2)
    right = rightamp * math.exp(-0.5 * (x.vector[0] - rightloc) ** 2 / rightsig ** 2)
    result = left + right
    return result


def officialGaussianForce(x):
    batch_size = 32
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result



def burgers_rkstiff_function(x):
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    return u0




v = CenteredGrid(burgers_rkstiff_function, **domain_dict)

physics = Burgers(default_viscosity=viscosity, diffusion_substeps=diffusion_substeps)

# for _ in view(play=False, namespace=globals()).range():
x = v.data._native.reshape(-1)  # initial state
V = []
t = [i for i in range(1, x.size + 1)]
for _ in range(40):
    v = physics.step(v, dt=dt)
    dt = 1.5 * dt
    if _ % 5 == 0:
        V.append(v.data._native.reshape(-1))
        # vis.show(v)
ax = waterfall(x, t, V, figsize=(8, 8))
ax.grid(False)
ax.axis(False)
ax.view_init(50, -100)
plt.show()
