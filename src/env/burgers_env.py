from typing import Tuple, List, Optional, Union

import gym
import numpy as np
import phi.flow as phiflow
from gym.wrappers.normalize import RunningMeanStd

from src.util.burgers_util import GaussianClash, GaussianForce
from src.visualization import LivePlotter


class BurgersEnv(gym.Env):

    def __init__(self,
                 step_count: int = 32,
                 domain: phiflow.Domain = phiflow.Domain((32,), box=phiflow.box[0:1]),
                 dt: float = 0.03,
                 viscosity: float = 0.003,
                 diffusion_substeps: int = 1,
                 exp_name: str = 'v0',
                 ):
        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        self.domain = domain
        self.exp_name = exp_name
        self.step_count = step_count
        self.dt = dt
        self.viscosity = viscosity
        self.physics = phiflow.Burgers(diffusion_substeps=diffusion_substeps)

        self.once_flag = False
        self.ep_idx = 0
        self.step_idx = 0
        self.init_state = None
        self.goal_state = None
        self.cont_state = None
        self.gt_state = None
        self.gt_forces = None

        self.reward_rms = RunningMeanStd()
        self.actions = None

        self.lviz = None

    def reset(self):
        self.step_idx = 0

        self.gt_forces = self._get_gt_forces()
        self.init_state = self._get_init_state()
        self.cont_state = self.init_state.copied_with()
        self.goal_state = self._get_goal_state()

        if self.once_flag:
            self.gt_state = self.init_state.copied_with()

        return self._build_obs()

    def step(self, action):
        self.step_idx += 1
        forces = self.actions
        forces_effect = phiflow.FieldEffect(phiflow.CenteredGrid(self.actions, box=self.domain.box), ['velocity'])
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # Perform reference simulation only when evaluating results -> after render was called once
        if self.once_flag:
            self.gt_state = self._step_gt()

        obs = self._build_obs()
        rew = self._build_rew(forces)
        done = np.full((1,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.ep_idx += 1

            missing_forces_field = (self.goal_state.velocity.data - self.cont_state.velocity.data) / self.dt
            missing_forces = phiflow.FieldEffect(phiflow.CenteredGrid(missing_forces_field, box=self.domain.box),
                                                 ['velocity'])
            forces += missing_forces_field
            self.cont_state = self.cont_state.copied_with(
                velocity=(self.cont_state.velocity.data + missing_forces_field * self.dt))

            add_rew = self._build_rew(missing_forces.field.data)
            rew += add_rew

            obs = self.reset()

        info = {'rew_unnormalized': rew[1], 'forces': np.abs(forces[1]).sum()}

        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)

        return obs, rew, done, info

    def render(self, mode: str = 'live') -> None:
        if not self.once_flag:
            self.once_flag = True
            self.gt_state = self.init_state.copied_with()
            if mode == 'live':
                self.lviz = LivePlotter()
            else:
                raise NotImplementedError()

        fields, labels = self._get_fields_and_labels()

        if mode == 'live':
            self.lviz.render(fields, labels, 2, True)
        else:
            raise NotImplementedError()

    def seed(self, seed: Optional[int] = None) -> Union[None, int]:
        return 42

    def close(self) -> None:
        pass

    def _build_obs(self) -> np.ndarray:
        curr_data = self.cont_state.velocity.data
        goal_data = self.goal_state.velocity.data

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_shape = curr_data.shape[1:-1] + (1,)
        time_data = np.full(curr_data.shape[1:], self.step_idx / self.step_count)
        # Channels last
        return np.array([np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)])

    @staticmethod
    def _build_rew(forces: np.ndarray) -> np.ndarray:
        reduced_shape = (forces.shape[0], -1)
        reshaped_forces = forces.reshape(reduced_shape)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    # The whole field with one parameter in each direction, flattened out
    @staticmethod
    def _get_act_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return act_dim,

    # Current and goal field with one parameter in each direction and one time channel
    @staticmethod
    def _get_obs_shape(field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)

    def _step_sim(self, in_state: phiflow.BurgersVelocity,
                  effects: Tuple[phiflow.FieldEffect, ...]) -> phiflow.BurgersVelocity:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _step_gt(self):
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _get_init_state(self) -> phiflow.BurgersVelocity:
        return phiflow.BurgersVelocity(domain=self.domain, velocity=GaussianClash(),
                                       viscosity=self.viscosity)

    def _get_goal_state(self) -> phiflow.BurgersVelocity:
        state = self.init_state.copied_with()
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        return state

    @staticmethod
    def _get_gt_forces() -> phiflow.FieldEffect:
        return phiflow.FieldEffect(GaussianForce(), ['velocity'])

    def _get_fields_and_labels(self) -> Tuple[List[np.ndarray], List[str]]:
        # Take the simulation of the first env
        fields = [f.velocity.data[0].reshape(-1) for f in [
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
