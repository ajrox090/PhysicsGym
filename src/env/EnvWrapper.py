import copy
from typing import Optional, Tuple, List

import gym
import numpy as np
from phi import math
from phi.field import Field
from phi.geom import Box
from phi.math import extrapolation, tensor
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.visualization import LivePlotter


class EnvWrapper(gym.Env):
    def __init__(self, ):
        super(EnvWrapper, self).__init__()

        # initialization and step variables
        self.N = 0
        self.dt = 0
        self.step_count = 0
        self.domain_dict = None

        # variables specific to problem
        self.observation_space = None
        self.action_space = None
        self.physics = None

        # states and forces
        self.gt_state = None
        self.gt_forces = None
        self.init_state = None
        self.goal_state = None
        self.cont_state = None
        self.trajectory = []

        self.actions = None
        self.ep_idx = 0
        self.step_idx = 0
        self.rendering = None
        self.test_mode = False
        self.lviz = LivePlotter("plots/")
        self.reward_range = (-float('inf'), float('inf'))

    def reset(self):
        # set and reset some variables
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        raise NotImplementedError

    def render(self, mode: str = 'live'):
        fields, labels = self._get_fields_and_labels()
        self.lviz.render(fields, labels, 5, True)
    #
    # def _build_obs(self):
    #     """build the output: observation based on observation space"""
    #     raise NotImplementedError
    #
    # def _build_reward(self, forces: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    #
    # def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    #     raise NotImplementedError
    #
    # def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    #     raise NotImplementedError

    def _step_gt(self):
        assert (self.gt_state is not None)
        assert (self.gt_forces is not None)
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _step_sim(self, in_state: Field,
                  effects: Tuple[FieldEffect, ...]) -> Field:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def disable_rendering(self):
        self.rendering = False

    def enable_rendering(self):
        self.rendering = True

    def _get_fields_and_labels(self) -> Tuple[List[np.ndarray], List[str]]:
        # Take the simulation of the first env
        fields = [f.data._native.reshape(-1) for f in [
            self.init_state,
            self.goal_state,
            self.gt_state,
            self.cont_state,
        ]]

        labels = [
            'Initial state',
            'Goal state',
            'Ground truth simulation',
            'Controlled simulation',
        ]

        return fields, labels

    def GaussianClash(self, x):
        batch_size = self.N
        leftloc = np.random.uniform(0.2, 0.4, batch_size)
        leftamp = np.random.uniform(0, 3, batch_size)
        leftsig = np.random.uniform(0.05, 0.15, batch_size)
        rightloc = np.random.uniform(0.6, 0.8, batch_size)
        rightamp = np.random.uniform(-3, 0, batch_size)
        rightsig = np.random.uniform(0.05, 0.15, batch_size)
        left = tensor(leftamp, x.shape[0]) * math.exp(
            -0.5 * (x.x.tensor - tensor(leftloc, x.shape[0])) ** 2 / tensor(leftsig, x.shape[0]) ** 2)
        right = tensor(rightamp, x.shape[0]) * math.exp(
            -0.5 * (x.x.tensor - tensor(rightloc, x.shape[0])) ** 2 / tensor(rightsig, x.shape[0]) ** 2)
        result = left + right
        return result

    def GaussianForce(self, x):
        batch_size = self.N
        loc = np.random.uniform(0.4, 0.6, batch_size)
        amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
        sig = np.random.uniform(0.1, 0.4, batch_size)
        result = tensor(amp, x.shape[0]) * math.exp(
            -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
        return result

    @staticmethod
    def ks_initial(x: math.Tensor):
        return math.cos(x) - 0.1 * math.cos(x / 16) * (1 - 2 * math.sin(x / 16))

    @staticmethod
    def ks_final(x: math.Tensor):
        return math.sin(x) - 0.1 * math.sin(x / 16) * (1 - 2 * math.cos(x / 16))