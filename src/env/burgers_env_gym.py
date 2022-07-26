import copy
import sys

import gym
import numpy as np
import phi.math
from phi import vis, math
from phi.geom import Box
from typing import Optional, Tuple, Union, Dict, List
from phi.math import extrapolation, instance, channel, tensor
from phi.field import Field, CenteredGrid
from phi.physics._effect import FieldEffect, GROW
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.phiflow.burgers import Burgers
from src.util.burgers_util import _get_obs_shape, _get_act_shape, _build_rew  # , GaussianForce, GaussianClash, \
# simpleGaussianClash, simpleGaussianForce
from src.visualization import LivePlotter

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class BurgersEnvGym(gym.Env):
    def __init__(self, N,
                 num_envs: int = 1,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 viscosity: float = 0.003,
                 diffusion_substeps: int = 1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0',
                 ):
        super(BurgersEnvGym, self).__init__()

        self.rendering = None
        if domain_dict is None:
            domain_dict = dict(extrapolation.PERIODIC, x=64, y=64, bounds=Box(x=200, y=100))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=_get_obs_shape(tuple([domain_dict['x']])))  # , domain_dict[
        # 'y']])))
        self.action_space = gym.spaces.Box(low=0, high=255, dtype=np.float32,
                                           shape=_get_act_shape(tuple([domain_dict['x']])))  # , domain_dict['y']])))

        self.N = N
        self.num_envs = num_envs
        self.reward_range = (-float('inf'), float('inf'))
        self.exp_name = exp_name
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.step_idx = 0
        self.ep_idx = 0
        self.dt = dt
        self.viscosity = viscosity
        self.physics = Burgers(diffusion_substeps=diffusion_substeps)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()
        self.actions = None
        self.test_mode = False
        self.init_state = None
        self.goal_state = None
        self.cont_state = None
        self.gt_state = None
        self.gt_forces = None
        self.lviz = None

    def reset(self) -> GymEnvObs:
        self.step_idx = 0

        self.gt_forces = self._get_gt_forces()
        self.init_state = self._get_init_state()
        self.cont_state = self.init_state
        self.goal_state = self._get_goal_state()
        if self.test_mode:
            self._init_ref_states()

        return self._build_obs()

    def step(self, actions: np.ndarray):
        self.actions = actions.reshape(self.cont_state.data._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.step_idx += 1
        forces = self.actions
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['velocity'])
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        if self.rendering:
            self.render()
        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.gt_state = self._step_gt()

        obs = self._build_obs()
        rew = _build_rew(forces)
        done = np.full((self.num_envs,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.ep_idx += 1

            missing_forces_field = (self.goal_state.data._native - self.cont_state.data._native) / self.dt
            missing_forces_field_tensor = phi.math.tensor(missing_forces_field, self.cont_state.shape)
            missing_forces = FieldEffect(CenteredGrid(missing_forces_field_tensor, **self.domain_dict),
                                         ['velocity'])
            forces += missing_forces_field
            self.cont_state = self.cont_state.data._native + missing_forces_field * self.dt

            add_rew = _build_rew(missing_forces.field.data._natives()[0]) * self.final_reward_factor
            rew += add_rew

            obs = self.reset()
        # normalize reward
        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        info = {'rew_unnormalized': rew, 'forces': np.abs(forces).sum()}
        # reward should be a single value not a list since it is just a single step in time.
        # assert len(rew) == 1
        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:

        if not self.test_mode:
            self.test_mode = True
            self._init_ref_states()
            self.lviz = LivePlotter("plots/")

        fields, labels = self._get_fields_and_labels()
        self.lviz.render(fields, labels, 2, True)
        # vis.show([self.cont_state, self.goal_state, self.gt_state])

    def disable_rendering(self):
        self.rendering = False

    def enable_rendering(self):
        self.rendering = True

    def _build_obs(self) -> np.ndarray:
        curr_data = copy.deepcopy(self.cont_state.data._native)
        goal_data = copy.deepcopy(self.goal_state.data._native)

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_data = np.full(curr_data.shape[1:-1] + (1,), self.step_idx / self.step_count)
        # Channels last
        return np.array([np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)])

    def _step_sim(self, in_state: Field,
                  effects: Tuple[FieldEffect, ...]) -> Field:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _step_gt(self):
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _get_init_state(self) -> Field:
        initState = CenteredGrid(self.GaussianClash, **self.domain_dict)
        return initState

    def _get_gt_forces(self) -> FieldEffect:
        return FieldEffect(CenteredGrid(self.GaussianForce, **self.domain_dict), ['velocity'])

    def _get_goal_state(self) -> Field:
        state = copy.deepcopy(self.init_state)
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        return state

    def _init_ref_states(self) -> None:
        self.gt_state = copy.deepcopy(self.init_state)

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
        batch_size = self.step_count
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
        batch_size = self.step_count
        loc = np.random.uniform(0.4, 0.6, batch_size)
        amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
        sig = np.random.uniform(0.1, 0.4, batch_size)
        result = tensor(amp, x.shape[0]) * math.exp(
            -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
        return result
