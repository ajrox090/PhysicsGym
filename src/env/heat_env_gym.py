import gym
import copy
import phi.math
import numpy as np
from phi.field import CenteredGrid
from phi.physics._effect import FieldEffect
from typing import Optional, Tuple, Union, Dict
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.EnvWrapper import EnvWrapper
from src.env.physics.heat import Heat

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class Heat1DEnvGym(EnvWrapper):
    def __init__(self,
                 N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 diffusivity: int = 0.1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0'):
        super(Heat1DEnvGym, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=self._get_obs_shape(tuple([domain_dict['x']])))  # , domain_dict[
        # 'y']])))
        self.action_space = gym.spaces.Box(low=0, high=255, dtype=np.float32,
                                           shape=self._get_act_shape(
                                               tuple([domain_dict['x']])))  # , domain_dict['y']])))

        self.N = N
        self.dt = dt
        self.exp_name = exp_name
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.diffusivity = diffusivity
        self.physics = Heat(diffusivity=diffusivity)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()

    def reset(self) -> GymEnvObs:
        self.step_idx = 0
        self.gt_forces = FieldEffect(CenteredGrid(self.GaussianForce, **self.domain_dict), ['temperature'])
        self.init_state = CenteredGrid(self.GaussianClash, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)
        # prepare goal state
        state = copy.deepcopy(self.init_state)
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        self.goal_state = state
        # init reference states
        if self.test_mode:
            self.gt_state = copy.deepcopy(self.init_state)
        return self._build_obs()

    def step(self, actions: np.ndarray):
        self.actions = actions.reshape(self.cont_state.data._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.step_idx += 1
        forces = self.actions
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['temperature_effect'])
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        if self.rendering:
            self.render()
        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.gt_state = self._step_gt()

        obs = self._build_obs()
        rew = self._build_reward(forces)
        done = np.full((1,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.ep_idx += 1

            missing_forces_field = (self.goal_state.data._native - self.cont_state.data._native) / self.dt
            missing_forces_field_tensor = phi.math.tensor(missing_forces_field, self.cont_state.shape)
            missing_forces = FieldEffect(CenteredGrid(missing_forces_field_tensor, **self.domain_dict),
                                         ['temperature_effect'])

            forces += missing_forces_field
            self.cont_state = self.cont_state.data._native + missing_forces_field * self.dt
            rew += self._build_reward(missing_forces.field.data._natives()[0]) * self.final_reward_factor
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
            self.gt_state = copy.deepcopy(self.init_state)
        super(Heat1DEnvGym, self).render()

    def _build_obs(self) -> np.ndarray:
        curr_data = copy.deepcopy(self.cont_state.data._native)
        goal_data = copy.deepcopy(self.goal_state.data._native)

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_shape = curr_data.shape[1:-1] + (1,)
        time_data = np.full(curr_data.shape[1:], self.step_idx / self.step_count)
        # Channels last
        return np.array([np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)])

    def _build_reward(self, forces: np.ndarray) -> np.ndarray:
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    # The whole field with one parameter in each direction, flattened out
    def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return act_dim,

    # Current and goal field with one parameter in each direction and one time channel
    def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)

