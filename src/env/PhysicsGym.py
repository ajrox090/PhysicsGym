import copy

import gym
import numpy as np
import phi
from phi import math
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from typing import Optional, Tuple

from phi.field import Field, CenteredGrid
from phi.math import tensor, inf
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd
from tqdm import tqdm


class PhysicsGym(gym.Env):
    def __init__(self,
                 domain, dx,
                 dt, step_count, domain_dict,
                 dxdt: int = 100,
                 reward_rms: [Optional] = RunningMeanStd()):
        super(PhysicsGym, self).__init__()

        # initialization and step variables
        self.dx = dx
        self.domain = domain
        self.N = int(domain / dx)
        self.dt = dt
        self.step_count = step_count
        self.domain_dict = domain_dict

        self.dxdt = dxdt  # number of time steps to simulate the environment
        self.observation_space = gym.spaces.Box(low=-inf, high=inf, dtype=np.float32,
                                                shape=self._get_obs_shape())
        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32,
                                           shape=self._get_act_shape())

        # variables specific to problem
        self.physics = None

        # states and forces
        self.init_state = None
        self.cont_state = None
        self.reference_state_np = None
        self.reference_state_phi = None

        self.ep_idx = 0
        self.step_idx = 0
        self.actions = None
        self.reward_rms = reward_rms
        self.reward_range = (-float('inf'), float('inf'))

        self._render = False

    def reset(self):
        """ set: - initial state, cont state and reference state
            returns: observation """
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        """ transform(x) := domain(d=1) -> domain(x)
            # 1) pre-processing: 1-d actions + transform(actions)
            # 2) update environment: step env
            # 3) post-processing:
            #       - transform(observations)
            #       - compute(rewards)"""
        raise NotImplementedError

    def render(self, mode: str = 'live'):
        raise NotImplementedError

    def step_physics(self, in_state: Field,
                     effects: Tuple[FieldEffect, ...]) -> Field:
        for i in range(self.dxdt):
            in_state = self.physics.step(in_state, dt=self.dt, effects=effects)
        return in_state

    def enable_rendering(self):
        self._render = True

    def disable_rendering(self):
        self._render = False

    def _get_obs_shape(self):
        return self.N,

    def _get_act_shape(self):
        return 1,

    def _build_obs(self):
        raise NotImplementedError

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def action_transform(self, alpha):
        # initialize a normal distribution with frozen in mean=-1, std. dev.= 1
        rv = norm(loc=0.5, scale=0.2)
        # rv = norm(loc=self.domain/2, scale=0.2)
        x = np.arange(0, self.domain, self.dx)
        return alpha * rv.pdf(x) / 2

    def scalar_action_to_forces(self, actions: np.ndarray):
        return FieldEffect(CenteredGrid(phi.math.tensor(self.action_transform(actions[0]).reshape(
            self.cont_state.data.native("x,vector").shape[0]), self.cont_state.shape), **self.domain_dict),
            ['temperature_effect'])

    @staticmethod
    def forces_to_numpy(forces: FieldEffect):
        return forces.field.data.native("vector,x")[0]

    def simpleUniformRandom(self, x):
        return tensor(-np.random.uniform(0, 0.5, self.N), x.shape[0])

    def simpleNormalDistribution(self, x):
        return tensor(norm(0, 0.1).pdf(x.native("vector,x")[0]) / 2, x.shape[0])

    def justOnes(self, x):
        return tensor([1.0 for _ in range(self.N)], x.shape[0])
