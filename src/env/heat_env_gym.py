import random
from typing import Tuple, Optional, List, Union, Any, Type, Dict

import gym
import numpy as np
import phi.flow as phiflow
import matplotlib.pyplot as plt
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.util.heat_util import GaussianClash, GaussianForce
from src.visualization import LivePlotter, GifPlotter

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class HeatEnvGym(gym.Env):
    def __init__(self,
                 N,
                 num_envs: int,
                 step_count: int = 32,
                 domain: phiflow.Domain = phiflow.Domain([50, ], box=phiflow.box[0:1]),
                 dt: float = 0.03,
                 default_diffusivity: int = 0.1,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0'):
        super(HeatEnvGym, self).__init__()

        act_shape = self._get_act_shape(domain.resolution)
        obs_shape = self._get_obs_shape(domain.resolution)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=act_shape, dtype=np.float32)

        self.N = N
        self.num_envs = num_envs
        self.reward_range = (-float('inf'), float('inf'))
        self.exp_name = exp_name
        self.domain = domain
        self.step_count = step_count
        self.step_idx = 0
        self.ep_idx = 0
        self.dt = dt
        self.default_diffusivity = default_diffusivity
        self.physics = phiflow.HeatDiffusion(default_diffusivity=default_diffusivity)
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
        self.gifviz = None
        self.pngviz = None
        self.vis_list = []
        self.temperature_goal_state = []

    def reset(self) -> GymEnvObs:
        self.step_idx = 0

        self.gt_forces = self._get_gt_forces()
        self.init_state = self._get_init_state()
        self.cont_state = self.init_state.copied_with()
        self.goal_state = self._get_goal_state()

        if self.test_mode:
            self._init_ref_states()

        return self._build_obs()

    def step(self, actions: np.ndarray):
        self.actions = actions.reshape(self.cont_state.temperature.data.shape)
        self.step_idx += 1
        forces = self.actions
        forces_effect = phiflow.FieldEffect(phiflow.CenteredGrid(self.actions, box=self.domain.box),
                                            ['temperature_effect'], mode=phiflow.GROW)
        # forces_effect = phiflow.HeatSource(self.actions, rate=self.dt,
        #                                     name='temperature_effect') # currently doesn't work, investigate.
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))
        self.vis_list.append(self.cont_state)

        self.render(mode='live')
        # Perform reference simulation only when evaluating results -> after render was called once
        if self.test_mode:
            self.gt_state = self._step_gt()
            self.temperature_goal_state.append(self.goal_state)

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

        info = {'rew_unnormalized': rew, 'forces': np.abs(forces).sum()}

        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)

        # reward should be a single value not a list since it is just a single step in time.
        assert len(rew) == 1
        return obs, rew[0], done, info

    def render(self, mode: str = 'live') -> None:
        if not self.test_mode:
            self.test_mode = True
            self._init_ref_states()
            if mode == 'live':
                self.lviz = LivePlotter("plots2/")
            elif mode == 'gif':
                self.gifviz = GifPlotter('StableHeatDiffusion-%s' % self.exp_name)
            else:
                raise NotImplementedError()

        fields, labels = self._get_fields_and_labels()

        if mode == 'live':
            self.lviz.render(fields, labels, 2, True)
        elif mode == 'gif':
            self.gifviz.render(fields, labels, 2, True, 'Temperature', self.ep_idx, self.step_idx, self.step_count,
                               True)
        elif mode == 'png':
            self.pngviz.render(fields, labels, 2, True, 'Velocity', self.ep_idx, self.step_idx, self.step_count, True)
        else:
            raise NotImplementedError()

    def _step_sim(self, temperature: phiflow.HeatTemperature,
                  effects: Tuple[phiflow.FieldEffect, ...]) -> phiflow.HeatTemperature:
        return self.physics.step(temperature, self.dt, effects=effects)

    def _step_gt(self):
        return self._step_sim(self.gt_state, (self.gt_forces,))

    def _get_init_state(self) -> phiflow.HeatTemperature:
        return phiflow.HeatTemperature(domain=self.domain, temperature=GaussianClash(self.num_envs),
                                       diffusivity=self.default_diffusivity)

    def _get_goal_state(self) -> phiflow.HeatTemperature:
        state = self.init_state.copied_with()
        for _ in range(self.step_count):
            state = self._step_sim(state, (self.gt_forces,))
        return state

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

    def _build_rew(self, forces: np.ndarray) -> np.ndarray:
        reduced_shape = (forces.shape[0], -1)
        reshaped_forces = forces.reshape(reduced_shape)
        # reduced_shape_forces = (self.goal_state.temperature.data.shape[0], -1)
        # reduced_gt_forces = self.goal_state.temperature.data.reshape(reduced_shape_forces)
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

    def show_state(self, title='all temperatures flow'):
        assert len(self.vis_list) > 0
        vels = [v.temperature.data.reshape(self.N, 1) for v in self.vis_list]  # gives a list of 2D arrays
        vels_img = np.array(np.concatenate(vels, axis=1), dtype=np.float32)

        # convert state's temperature data to an image
        # state_img = np.asarray(np.concatenate(instate.temperature.data, axis=-1), dtype=np.float32)
        # we only have 33 time steps, blow up by a factor of 2^4 to make it easier to see
        # (could also be done with more evaluations of network)
        state_img = np.expand_dims(vels_img, axis=2)
        for i in range(4):
            state_img = np.concatenate([state_img, state_img], axis=2)

        state_img = np.reshape(state_img, [state_img.shape[0], state_img.shape[1] * state_img.shape[2]])
        # print("Resulting image size" + format(state_img.shape))

        fig, axes = plt.subplots(1, 1, figsize=(16, 5))
        im = axes.imshow(state_img, origin='upper', cmap='inferno')
        plt.colorbar(im)
        plt.xlabel('time')
        plt.ylabel('x')
        plt.title(title)
        plt.show()

    def show_vels(self, title='state of all temperatures'):

        assert len(self.temperature_goal_state) > 0
        vels = [v.temperature.data.reshape(self.N, 1) for v in self.vis_list]  # gives a list of 2D arrays
        vels_goal = [v.temperature.data.reshape(self.N, 1) for v in
                     self.temperature_goal_state]  # gives a list of 2D arrays
        fig = plt.figure().gca()
        cmap = plt.cm.get_cmap('hsv', self.step_count)  # list 1D

        def random_int():
            return random.randint(0, 100)

        color = random_int()
        label = "t"
        for i in range(self.step_count):
            a1 = self.step_count // 3
            a2 = (self.step_count * 2) // 3
            # choose random colors for each 1/3 of step_count
            if (i % a1 == 0) or (i % a2 == 0):
                label = "t" + str(i)
                color = cmap(random_int())

            fig.plot(np.linspace(-1, 1, len(vels[i].flatten())), vels[i].flatten(),
                     lw=1, color=color, label=label if (i % a1 == 0) or (i % a2 == 0) else "")
        fig.plot(np.linspace(-1, 1, len(vels_goal[self.step_count].flatten())), vels_goal[self.step_count].flatten(),
                 lw=2, color='gray', label="goal_state")

        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.title(title)
        plt.show()