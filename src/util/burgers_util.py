from typing import Tuple

from matplotlib import pyplot as plt
from phi import math
import numpy as np
from phi.math import tensor


def SimpleGaussian(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2 / 0.34 ** 2)
    right = math.exp(-0.5 * (x - 0.34) ** 2 / 0.34 ** 2)
    result = left + right
    return result


def simpleGaussianForce(x):
    return 2.4 * math.exp(-0.5 * (x - 0.34) ** 2 / 0.34 ** 2)


def simpleGaussianClash(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2)
    right = 2.4 * math.exp(-0.5 * (x - 0.34) ** 2)
    result = left + right
    return result


def GaussianClash(x):
    leftloc = np.random.uniform(0.2, 0.4)
    leftamp = np.random.uniform(0, 3)
    leftsig = np.random.uniform(0.05, 0.15)
    rightloc = np.random.uniform(0.6, 0.8)
    rightamp = np.random.uniform(-3, 0)
    rightsig = np.random.uniform(0.05, 0.15)
    left = tensor(leftamp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(leftloc, x.shape[0])) ** 2 / tensor(leftsig, x.shape[0]) ** 2)
    right = tensor(rightamp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(rightloc, x.shape[0])) ** 2 / tensor(rightsig, x.shape[0]) ** 2)
    result = left + right
    return result


def GaussianForce(x):
    batch_size = 32
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result


def waterfall(x, t, u, **kwargs):
    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs['figsize'])
    else:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    for i, snapshot in enumerate(u):
        ax.plot(x, t[i] * np.ones_like(x), snapshot,
                color=np.random.choice(['blue', 'black']))
    plt.xlim([x[0], x[-1]])
    plt.ylim([t[0], t[-1]])
    plt.tight_layout()
    return ax
