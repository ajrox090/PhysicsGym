import copy
import math

import matplotlib.pyplot as plt
from phi.flow import *
from phi.physics._effect import FieldEffect

from src.env.physics.ks import KuramotoSivashinsky

N = 20
domain_dict = dict(x=N, bounds=Box(x=20), extrapolation=extrapolation.PERIODIC)
step_count = 1
dt = 0.01


def officialGaussianClash(x):
    batch_size = 20
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
    batch_size = 20
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result


def simpleSine(x):
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    # u0 = math.sin(x)
    return u0


def simpleCosine(x):
    u0 = math.exp(-10 * math.cos(x / 2) ** 2)
    # u0 = math.cos(x)
    return u0


# physics = KuramotoSivashinsky()
# gt_forces = FieldEffect(CenteredGrid(simpleCosine, **domain_dict), ['velocity'])
# init_state = CenteredGrid(simpleSine, **domain_dict)
# cont_state = copy.deepcopy(init_state)
# # prepare goal state
# state = copy.deepcopy(init_state)
# for _ in view('hbb', play=False, framerate=10, namespace=globals()).range():
#     # state = physics.step(state, dt=dt, dependend_states=(gt_forces,))
#     state = physics.step(state, dt=dt)
#
# goal_state = state
#
# # for _ in view('u', play=False, framerate=10, namespace=globals()).range():
#     # for _ in range(300):
#     u = physics.step(u, dt=0.01)
#     ux = u.values.native('x,vector')
#
#     # plt.plot(ux)
#     # ax = plt.gca()
#     # ax.set_xlim([-5, N])
#     # ax.set_ylim([-5, N])
# plt.show()
# # vis.show(u)
