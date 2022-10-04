import copy
from typing import Optional, Tuple, List

import gym
import numpy as np
import phi
from phi import math, vis
from phi.field import Field, CenteredGrid
from phi.geom import Box
from phi.math import extrapolation, tensor
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.physics.heat import Heat
from src.visualization import LivePlotter


class PhysicsGym(gym.Env):
    def __init__(self, ):
        super(PhysicsGym, self).__init__()

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
        """ set:
                - initial state and goal state"""
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


class TestPhysicsGym(PhysicsGym):
    def __init__(self,
                 N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 diffusivity: int = 0.1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None):
        super(TestPhysicsGym, self).__init__()

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
        self.physics = Heat(diffusivity=diffusivity)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = RunningMeanStd()

    def reset(self):
        self.step_idx = 0
        self.init_state = CenteredGrid(self.GaussianClash, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)

        # self.goal_state = CenteredGrid(np.zeros(self.N), **self.domain_dict)
        self.goal_state = np.zeros(self.N).reshape(self.N, 1)
        self.phi_goal_state = CenteredGrid(self.goal_state.reshape(-1), **self.domain_dict)
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
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['temperature_effect'])

        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # if self.step_idx % 20 == 0:
        self.render()

        obs = self._build_obs()
        rew = self._build_reward(obs, forces)
        done = np.full((1,), self.step_idx == self.step_count)

        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        info = {'rew_unnormalized': rew, 'forces': np.abs(forces).sum()}
        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        vis.show(self.cont_state, self.phi_goal_state)

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data._native.reshape(-1)

    def _build_reward(self, obs: np.ndarray, forces: np.ndarray) -> np.ndarray:
        obs_mse = -np.sum((obs - self.goal_state)**2 / self.N, axis=-1)
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        sum_forces = -np.sum(reshaped_forces ** 2, axis=-1)
        return obs_mse + sum_forces

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self) -> Tuple[int, ...]:
        # act_dim = np.prod(field_shape) * len(field_shape)
        return 1,

    def _get_obs_shape(self) -> Tuple[int, ...]:
        return tuple(self.N, )
