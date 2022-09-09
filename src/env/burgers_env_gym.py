import gym
import copy
import phi.math
import numpy as np
from matplotlib import pyplot as plt, colors
from phi import field, vis
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import spatial
from phi.physics._effect import FieldEffect
from typing import Optional, Tuple, Union, Dict
from stable_baselines3.common.running_mean_std import RunningMeanStd
from tqdm import tqdm

from src.env.EnvWrapper import EnvWrapper
from src.env.physics.burgers import Burgers

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class Burgers1DEnvGym(EnvWrapper):
    def __init__(self, N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 viscosity: float = 0.003,
                 diffusion_substeps: int = 1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0',
                 ):
        super(Burgers1DEnvGym, self).__init__()

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
        self.viscosity = viscosity
        self.diffusion_substeps = diffusion_substeps
        self.physics = Burgers(diffusion_substeps=diffusion_substeps)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()
        self.goal_trajectory = []
        self.raw_trajectory = []
        self.raw_state = None

    def reset(self) -> GymEnvObs:
        self.step_idx = 0
        self.goal_trajectory = []
        self.physics = Burgers(diffusion_substeps=self.diffusion_substeps)
        self.gt_forces = FieldEffect(CenteredGrid(self.GaussianForce, **self.domain_dict), ['velocity'])
        self.init_state = CenteredGrid(self.GaussianClash, **self.domain_dict)
        self.raw_trajectory = []
        self.cont_state = copy.deepcopy(self.init_state)
        # calculate goal state based on ground truth forces
        state = copy.deepcopy(self.init_state)
        raw_state = copy.deepcopy(self.init_state)
        for _ in tqdm(range(self.step_count)):
            state = self._step_sim(state.vector['x'], (self.gt_forces,))
            raw_state = self._step_sim(raw_state.vector['x'], ())
            self.goal_trajectory.append(state.vector['x'])
            self.raw_trajectory.append(raw_state.vector['x'])
        self.goal_state = state
        self.raw_state = self.raw_trajectory[-1]
        # init reference states
        if self.test_mode:
            self.gt_state = copy.deepcopy(self.init_state)
        return self._build_obs()

    def step(self, actions: np.ndarray):
        self.actions = actions.reshape(self.cont_state.data._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.step_idx += 1
        forces = self.actions
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['velocity_effect'])
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        if self.rendering:
            if self.step_idx % 20 == 0:
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
                                         ['velocity'])
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
        super().render()

    def gradient_color(self, frame, frame_count, cmap='RdBu'):
        import matplotlib.cm as cmx
        jet = plt.get_cmap(cmap)
        cNorm = colors.Normalize(vmin=0, vmax=frame_count)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        return scalarMap.to_rgba(frame)

    def last_render(self):
        rl_frames = self.physics.trajectory
        gt_frames = self.goal_trajectory
        unc_frames = self.raw_trajectory
        fig, axs = plt.subplots(1, 3, figsize=(18.9, 9.6))
        axs[0].set_title("Reinforcement Learning")
        axs[1].set_title("Ground Truth")
        axs[2].set_title("Uncontrolled")
        for plot in axs:
            plot.set_ylim(-2, 2)
            plot.set_xlabel('x')
            plot.set_ylabel('u(x)')

        for frame in range(0, self.step_count):
            frame_color = self.gradient_color(frame, self.step_count)
            axs[0].plot(rl_frames[frame].data.native('vector,x')[0], color=frame_color, linewidth=0.8)
            axs[1].plot(gt_frames[frame].data.native('vector,x')[0], color=frame_color, linewidth=0.8)
            axs[2].plot(unc_frames[frame].data.native('vector,x')[0], color=frame_color, linewidth=0.8)
        plt.show()
        # render state trajectory
        # temp_t = field.stack(self.physics.trajectory, spatial('time'),
        #                      Box(time=len(self.physics.trajectory) * self.dt))  # time=len(trajectory)* dt
        # vis.show(temp_t.vector[0], aspect='auto', size=(8, 3), title='optimal state trajectory')
        # # render goal state trajectory
        # temp_gt = field.stack(self.goal_trajectory, spatial('time'), Box(time=len(self.goal_trajectory) * self.dt))
        # vis.show(temp_gt.vector[0], aspect='auto', size=(8, 3), title='goal state trajectory')
        # # raw state trajectory
        # temp_rt = field.stack(self.raw_trajectory, spatial('time'), Box(time=len(self.raw_trajectory) * self.dt))
        # vis.show(temp_rt.vector[0], aspect='auto', size=(8, 3), title='raw state trajectory')

    def _build_obs(self) -> np.ndarray:
        curr_data = copy.deepcopy(self.cont_state.data._native)
        goal_data = copy.deepcopy(self.goal_state.data._native)

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_data = np.full(curr_data.shape[1:-1] + (1,), self.step_idx / self.step_count)
        # Channels last
        return np.array([np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)])

    def _build_reward(self, forces: np.ndarray) -> np.ndarray:
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return act_dim,

    def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)


class Burgers2DEnvGym(EnvWrapper):
    def __init__(self, N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 viscosity: float = 0.003,
                 diffusion_substeps: int = 1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0',
                 ):
        super(Burgers2DEnvGym, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=self._get_obs_shape(tuple([domain_dict['x'], domain_dict['y']])))
        self.action_space = gym.spaces.Box(low=0, high=255, dtype=np.float32,
                                           shape=self._get_act_shape(
                                               tuple([domain_dict['x'], domain_dict['y']])))

        self.N = N
        self.dt = dt
        self.exp_name = exp_name
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.viscosity = viscosity
        self.physics = Burgers(diffusion_substeps=diffusion_substeps)
        self.final_reward_factor = final_reward_factor
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()

    def reset(self) -> GymEnvObs:
        self.step_idx = 0
        self.gt_forces = FieldEffect(CenteredGrid(self.GaussianForce, **self.domain_dict), ['velocity'])
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
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['velocity_effect'])
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
                                         ['velocity'])
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
        super(Burgers2DEnvGym, self).render()

    def _build_obs(self) -> np.ndarray:
        curr_data = copy.deepcopy(self.cont_state.data._native)
        goal_data = copy.deepcopy(self.goal_state.data._native)

        # Preserve the spacial dimensions, cut off batch dim and use only one channel
        time_data = np.full(curr_data.shape[1:-1] + (1,), self.step_idx / self.step_count)
        # Channels last
        return np.array([np.concatenate(obs + (time_data,), axis=-1) for obs in zip(curr_data, goal_data)])

    def _build_reward(self, forces: np.ndarray) -> np.ndarray:
        reshaped_forces = forces.reshape(forces.shape[0], -1)
        return -np.sum(reshaped_forces ** 2, axis=-1)

    def _get_act_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        act_dim = np.prod(field_shape) * len(field_shape)
        return act_dim,

    def _get_obs_shape(self, field_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(field_shape) + (2 * len(field_shape) + 1,)
