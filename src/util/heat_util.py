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
    left = tensor(leftamp, x.shape[0]) * math.exp(-0.5 * (x.x.tensor - tensor(leftloc, x.shape[0])) ** 2 / tensor(leftsig, x.shape[0]) ** 2)
    right = tensor(rightamp, x.shape[0]) * math.exp(-0.5 * (x.x.tensor - tensor(rightloc, x.shape[0])) ** 2 / tensor(rightsig, x.shape[0]) ** 2)
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