from typing import Tuple

import numpy as np


def _build_rew(forces: np.ndarray) -> np.ndarray:
    reshaped_forces = forces.reshape(forces.shape[0], -1)
    return -np.sum(reshaped_forces ** 2, axis=-1)


def _get_act_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    act_dim = np.prod(field_shape) * len(field_shape)
    return act_dim,


def _get_obs_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(field_shape) + (2 * len(field_shape) + 1,)


# todo: Pending
def GaussianClashFunction(x):
    batch_size = 32
    idx = np.array(x.vector[0])
    leftloc = np.random.uniform(0.2, 0.4, batch_size)
    leftamp = np.random.uniform(0, 3, batch_size)
    leftsig = np.random.uniform(0.05, 0.15, batch_size)
    rightloc = np.random.uniform(0.6, 0.8, batch_size)
    rightamp = np.random.uniform(-3, 0, batch_size)
    rightsig = np.random.uniform(0.05, 0.15, batch_size)
    # idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
    left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
    right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
    result = left + right
    result = np.swapaxes(result, 0, -1)
    return result