from typing import Tuple
from phi import math
import numpy as np


def _build_rew(forces: np.ndarray) -> np.ndarray:
    reshaped_forces = forces.reshape(forces.shape[0], -1)
    return -np.sum(reshaped_forces ** 2, axis=-1)


def _get_act_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    act_dim = np.prod(field_shape) * len(field_shape)
    return act_dim,


def _get_obs_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(field_shape) + (2 * len(field_shape) + 1,)


def SimpleGaussian(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2)
    right = math.exp(-0.5 * (x - 0.34) ** 2)
    result = left + right
    return result


def GaussianForce(x):
    return 3.4 * math.exp(-0.5 * (x - 0.34) ** 2 / 0.34 ** 2)


def GaussianClash(x):
    left = math.exp(-0.5 * (x - 0.09) ** 2)
    right = 2.4 * math.exp(-0.5 * (x - 0.34) ** 2)
    result = left + right
    return result
