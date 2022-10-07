import copy
from typing import Optional, Tuple, List

import gym
import numpy as np
import phi
from matplotlib import pyplot as plt
from phi import math, vis
from phi.field import Field, CenteredGrid
from phi.geom import Box
from phi.math import extrapolation, tensor, inf
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from scipy.stats import norm
from src.env.physics.burgers import Burgers
from src.env.physics.heat import Heat
from src.visualization import LivePlotter


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


class HeatPhysicsGym(PhysicsGym):
    def __init__(self,
                 domain: int = 5,
                 dx: float = 0.25,
                 step_count: int = 1000,
                 domain_dict=None,
                 dt: float = 0.01,
                 diffusivity: int = 0.3,
                 final_reward_factor: float = 32):
        super(HeatPhysicsGym, self).__init__(domain, dx, dt, step_count, domain_dict,
                                             final_reward_factor, reward_rms=RunningMeanStd())

        self.initial_state = None
        self.final_state = None
        self.actions_grid_trans = None
        self.diffusivity = diffusivity
        self.physics = Heat(diffusivity=diffusivity)

    def reset(self):
        self.step_idx = 0

        self.init_state = CenteredGrid(self.simpleGauss, **self.domain_dict)
        if self.initial_state is None:
            self.initial_state = copy.deepcopy(self.init_state)
        # if self.cont_state is None:
        #     self.cont_state = copy.deepcopy(self.init_state)
        self.cont_state = copy.deepcopy(self.init_state)

        self.reference_state_np = np.zeros(self.N).reshape(self.N, 1)
        self.reference_state_phi = CenteredGrid(self.reference_state_np.reshape(-1)[0], **self.domain_dict)
        return self._build_obs()

    def step(self, actions: np.ndarray):

        # prepare actions
        actions = self.action_transform(actions[0])
        self.actions = actions.reshape(self.cont_state.data.native("x,vector").shape[0])
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.actions_grid_trans = CenteredGrid(actions_tensor, **self.domain_dict)
        forces_effect = FieldEffect(self.actions_grid_trans, ['temperature_effect'])

        # step environment
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # visualize
        if self.step_idx % (self.step_count-1) == 0:
            self.render()

        # post-processing
        self.step_idx += 1
        obs = self._build_obs()
        rew = self._build_reward(obs)
        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        done = np.full((1,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.final_state = copy.deepcopy(self.cont_state)
        info = {'rew_unnormalized': rew}

        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        x = np.arange(0, self.domain, self.dx)
        plt.tick_params(axis='x', which='minor', length=10)
        plt.grid(True, linestyle='--', which='both')
        plt.plot(x, self.init_state.data.native("vector,x")[0], label='init state')
        plt.plot(x, self.actions_grid_trans.data.native("vector,x")[0], label='action')
        plt.plot(x, self.cont_state.data.native("vector,x")[0], label='cont state')
        plt.plot(x, self.reference_state_np, label='final state')
        plt.xlim(0, self.domain)
        plt.ylim(-3, 3)
        plt.legend()
        plt.show()

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data.native("vector,x")[0]

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        return -np.sum((obs - self.reference_state_np) ** 2 / self.N, axis=-1)


class TestPhysicsGymBurgers(PhysicsGym):
    def __init__(self,
                 N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 diffusivity: int = 0.1,
                 final_reward_factor: float = 32):
        super(TestPhysicsGymBurgers, self).__init__()

        self.actions_grid_trans = None
        self.phi_goal_state = None
        self.observation_space = gym.spaces.Box(low=-N, high=N, dtype=np.float32,
                                                shape=(domain_dict['x'],))
        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32,
                                           shape=self._get_act_shape())

        self.N = N
        self.dt = dt
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.diffusivity = diffusivity
        self.physics = Burgers(default_viscosity=diffusivity)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = RunningMeanStd()

    def reset(self):
        self.step_idx = 0
        self.init_state = CenteredGrid(self.GaussianClash, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)

        self.goal_state = np.zeros(self.N).reshape(self.N, 1)
        u = copy.deepcopy(self.init_state)
        physics = Burgers(default_viscosity=self.diffusivity)
        for i in range(self.step_count):
            u = physics.step(u, dt=self.dt)
        self.phi_goal_state = u
        # self.phi_goal_state = CenteredGrid(self.goal_state.reshape(-1), **self.domain_dict)
        # vis.show(self.init_state, self.phi_goal_state)
        return self._build_obs()

    def step(self, actions: np.ndarray):
        # transform
        def transformx(alpha):
            a = [alpha * np.sin(x) for x in range(self.N)]
            return np.array(a)

        actions = transformx(actions[0])
        self.actions = actions.reshape(self.cont_state.data._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.step_idx += 1
        forces = self.actions
        self.actions_grid_trans = CenteredGrid(actions_tensor, **self.domain_dict)
        forces_effect = FieldEffect(self.actions_grid_trans, ['temperature_effect'])

        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        if self.step_idx % 20 == 0:
            self.render()

        obs = self._build_obs()
        rew = self._build_reward(obs, forces)
        done = np.full((1,), self.step_idx == self.step_count)

        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        info = {'rew_unnormalized': rew, 'forces': np.abs(forces).sum()}
        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        plt.plot(self.actions_grid_trans.data.native("vector,x")[0], label='action')
        plt.plot(self.cont_state.data.native("vector,x")[0], label='cont state')
        plt.plot(self.phi_goal_state.data.native("vector,x")[0], label='final state')
        plt.xlim(0, 8)
        plt.ylim(-3, 3)
        plt.legend()
        plt.show()
        # vis.show(self.cont_state, self.phi_goal_state, self.actions_grid_trans)

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data._native.reshape(-1)

    def _build_reward(self, obs: np.ndarray, forces: np.ndarray) -> np.ndarray:
        obs_mse = -np.sum((obs - self.goal_state) ** 2 / self.N, axis=-1)
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        sum_forces = -np.sum(reshaped_forces ** 2, axis=-1)
        return obs_mse + sum_forces

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self) -> Tuple[int, ...]:
        # act_dim = np.prod(field_shape) * len(field_shape)
        return 1,

    def _get_obs_shape(self) -> Tuple[int, ...]:
        return tuple(self.N, )
