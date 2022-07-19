
import gym
import numpy as np
import phi.math
from phi import math
from phi.geom import Box
from typing import Optional, Tuple, Union, Dict
from phi.math import extrapolation, instance, channel
from phi.field import Field, CenteredGrid
from phi.physics._effect import FieldEffect, GROW
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.phiflow.burgers import Burgers
from src.util.burgers_util import _get_obs_shape, _get_act_shape, _build_rew

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

        if domain_dict is None:
            domain_dict = dict(extrapolation.PERIODIC, x=64, y=64, bounds=Box(x=200, y=100))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=_get_obs_shape(tuple([domain_dict['x'], domain_dict['y']])))
        self.action_space = gym.spaces.Box(low=0, high=255, dtype=np.float32,
                                           shape=_get_act_shape(tuple([domain_dict['x'], domain_dict['y']])))

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
        self.actions = actions.reshape(self.cont_state.points._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.step_idx += 1
        forces = self.actions
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['velocity'],
                                    mode=GROW)
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.gt_state = self._step_gt()

        obs = self._build_obs()
        rew = _build_rew(forces)
        done = np.full((self.num_envs,), self.step_idx == self.step_count)
        # normalize reward
        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        info = {'rew_unnormalized': rew, 'forces': np.abs(forces).sum()}
        # reward should be a single value not a list since it is just a single step in time.
        # assert len(rew) == 1
        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        pass

    def _build_obs(self) -> np.ndarray:
        curr_data = self.cont_state.points._native
        goal_data = self.goal_state.points._native

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
        return CenteredGrid(lambda x: math.sin(x), **self.domain_dict)

    def _get_gt_forces(self) -> FieldEffect:
        return FieldEffect(CenteredGrid(lambda x: (math.cos(x) + math.sin(x)), **self.domain_dict), ['velocity'])

    def _get_goal_state(self) -> Field:
        state = self.init_state
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        return state

    def _init_ref_states(self) -> None:
        self.gt_state = self.init_state
