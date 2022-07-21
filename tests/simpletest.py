""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""

from phi.flow import *
from phi import math


def GaussianForce(x):
    batch_size = 63
    xshape = x.shape
    leftloc = np.random.uniform(0.2, 0.4, batch_size)
    leftamp = np.random.uniform(0, 3, batch_size)
    leftsig = np.random.uniform(0.05, 0.15, batch_size)
    rightloc = np.random.uniform(0.6, 0.8, batch_size)
    rightamp = np.random.uniform(-3, 0, batch_size)
    rightsig = np.random.uniform(0.05, 0.15, batch_size)
    # idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
    left = leftamp * math.exp(-0.5 * (x - leftloc) ** 2 / leftsig ** 2)
    right = rightamp * math.exp(-0.5 * (x - rightloc) ** 2 / rightsig ** 2)
    result = left + right
    # result = np.swapaxes(result, 0, -1)
    return 3.4 * math.exp(-0.5 * (x - 0.34) ** 2 / 0.34 ** 2)


def GaussianClash(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2)
    right = 2.4 * math.exp(-0.5 * (x - 0.34) ** 2)
    result = left + right
    return result


# DOMAIN = dict(x=64, y=64, bounds=Box(x=2 * math.pi, y=2 * math.pi), extrapolation=extrapolation.PERIODIC)
DOMAIN = dict(x=64, bounds=Box[0:1])
dt = 0.1
sim_velocity = CenteredGrid(GaussianForce, **DOMAIN)  # also works with CenteredGrid
# vis.plot(sim_velocity)
# vis.show()

# viewer = view(sim_velocity, namespace=globals(), play=False)
# for _ in viewer.range():
#     sim_velocity = diffuse.explicit(sim_velocity, diffusivity=0.01, dt=dt)
#     sim_velocity = advect.semi_lagrangian(sim_velocity, sim_velocity, dt)
