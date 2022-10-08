import gym
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple

from phi.field import Field
from phi.math import tensor, inf
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd


class PhysicsGym(gym.Env):
    def __init__(self,
                 domain, dx,
                 dt, step_count, domain_dict,
                 final_reward_factor, reward_rms: [Optional] = RunningMeanStd()):
        super(PhysicsGym, self).__init__()

        # initialization and step variables
        self.dx = dx
        self.domain = domain
        self.N = int(domain / dx)
        self.dt = dt
        self.step_count = step_count
        self.domain_dict = domain_dict

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
        self.final_reward_factor = final_reward_factor
        self.reward_range = (-float('inf'), float('inf'))

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

    def _step_sim(self, in_state: Field,
                  effects: Tuple[FieldEffect, ...]) -> Field:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

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
        x = np.arange(0, self.domain, self.dx)
        return alpha * rv.pdf(x) / 2

    def simpleGauss(self, x):
        return tensor(np.random.uniform(0, 0.5, self.N), x.shape[0])
