from typing import Tuple, Optional, List, Union, Any, Type

import numpy as np
import phi.flow as phiflow
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv
import gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
from phi.physics.heat import HeatDiffusion
from src.util.heat_util import GaussianClash, GaussianForce
from src.visualization import LivePlotter, GifPlotter


class HeatInvaderEnv(VecEnv):

    def __init__(self, num_envs: int,
                 step_count: int = 50,
                 diffusivity: int = 0.1,
                 dt: float = 0.03,
                 domain: phiflow.Domain = phiflow.Domain([50, ], box=phiflow.box[0:1]),
                 reward_rms: Optional[RunningMeanStd] = None,
                 final_reward_factor: float = 50,
                 exp_name: str = 'v0'):
        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)

        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.reward_range = (-float('inf'), float('inf'))
        self.step_count = step_count
        self.domain = domain
        self.exp_name = exp_name
        self.dt = dt
        self.diffusivity = diffusivity
        self.physics = phiflow.HeatDiffusion(diffusivity=diffusivity)
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()
        self.final_reward_factor = final_reward_factor

        self.init_state = None
        self.goal_state = None
        self.cont_state = None
        self.gt_state = None
        self.gt_forces = None
        self.actions = None
        self.step_idx = 0
        self.test_mode = False
        self.ep_idx = 0

        self.lviz = None
        self.gifviz = None

    def reset(self) -> VecEnvObs:
        self.step_idx = 0

        self.init_state = self._get_init_state()
        self.cont_state = self.init_state.copied_with()
        self.gt_forces = self._get_gt_forces()
        self.goal_state = self._get_goal_state()

        if self.test_mode:
            self._init_ref_states()

        return self._build_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions.reshape(self.cont_state.temperature.data.shape)

    def step_wait(self) -> VecEnvStepReturn:
        self.step_idx += 1
        forces = self.actions
        forces_effect = phiflow.FieldEffect(phiflow.CenteredGrid(self.actions, box=self.domain.box), ['temperature_effect'])
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.gt_state = self._step_gt()

        obs = self._build_obs()
        rew = self._build_rew(forces)
        done = np.full((self.num_envs,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.ep_idx += 1

            missing_forces_field = (self.goal_state.temperature.data - self.cont_state.temperature.data) / self.dt
            missing_forces = phiflow.FieldEffect(phiflow.CenteredGrid(missing_forces_field, box=self.domain.box),
                                                 ['temperature_effect'])
            forces += missing_forces_field
            self.cont_state = self.cont_state.copied_with(
                temperature=(self.cont_state.temperature.data + missing_forces_field * self.dt))

            add_rew = self._build_rew(missing_forces.field.data) * self.final_reward_factor
            rew += add_rew

            obs = self.reset()

        info = [{'rew_unnormalized': rew[i], 'forces': np.abs(forces[i]).sum()} for i in range(self.num_envs)]

        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)

        return obs, rew, done, info

    def close(self) -> None:
        pass

    def disable_test_mode_wtf(self):
        self.test_mode = False

    def render(self, mode: str = 'live') -> None:
        if not self.test_mode:
            self.test_mode = True
            self._init_ref_states()
            if mode == 'live':
                self.lviz = LivePlotter()
            elif mode == 'gif':
                self.gifviz = GifPlotter('StableBurger-%s' % self.exp_name)
            else:
                raise NotImplementedError()

        fields, labels = self._get_fields_and_labels()

        if mode == 'live':
            self.lviz.render(fields, labels, 2, True)
        elif mode == 'gif':
            self.gifviz.render(fields, labels, 2, True, 'Temperature', self.ep_idx, self.step_idx, self.step_count,
                               True)
        else:
            raise NotImplementedError()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        return [getattr(self, attr_name) for _ in self._vec_env_indices_to_list(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None):
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        getattr(self, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in self._vec_env_indices_to_list(indices)]

    def _step_sim(self, temperature: phiflow.HeatTemperature,
                  effects: Tuple[phiflow.FieldEffect, ...]) -> phiflow.HeatTemperature:
        return self.physics.step(temperature, self.dt, effects=effects)

    def _step_gt(self):
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _get_goal_state(self) -> phiflow.HeatTemperature:
        state = self.init_state.copied_with()
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        return state

    def _get_init_state(self) -> phiflow.HeatTemperature:
        return phiflow.HeatTemperature(domain=self.domain, temperature=GaussianClash(self.num_envs),
                                       diffusivity=self.diffusivity)

    def _get_gt_forces(self) -> phiflow.FieldEffect:
        return phiflow.FieldEffect(GaussianForce(self.num_envs), ['temperature'])

    def _init_ref_states(self) -> None:
        self.gt_state = self.init_state.copied_with()

    def _build_obs(self) -> np.ndarray:
        curr_data = self.cont_state.temperature.data
        goal_data = self.goal_state.temperature.data

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

    @staticmethod
    def _vec_env_indices_to_list(raw_indices: VecEnvIndices) -> List[int]:
        if raw_indices is None:
            return []
        if isinstance(raw_indices, int):
            return [raw_indices]
        return list(raw_indices)

    def _get_fields_and_labels(self) -> Tuple[List[np.ndarray], List[str]]:
        # Take the simulation of the first env
        fields = [f.temperature.data[0].reshape(-1) for f in [
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
