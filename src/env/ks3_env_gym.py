import gym
import copy
import phi.math
import numpy as np
from phi import field, vis
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import spatial, extrapolation
from phi.physics._effect import FieldEffect
from typing import Optional, Tuple, Union, Dict
from stable_baselines3.common.running_mean_std import RunningMeanStd
from tqdm import tqdm

from src.env.EnvWrapper import EnvWrapper
from src.env.physics.ks3 import KuramotoSivashinsky
from src.util.ks_util import ks_initial, ks_initial2
from tests.simple_ks_simulation import simpleSine, simpleCosine

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class KS3EnvGym(EnvWrapper):
    def __init__(self, N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 exp_name: str = 'v0',
                 ):
        super(KS3EnvGym, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=self._get_obs_shape(tuple([domain_dict['x']])))
        self.action_space = gym.spaces.Box(low=-1, high=0, dtype=np.float32,
                                           shape=self._get_act_shape(tuple([domain_dict['x']])))

        self.N = N
        self.dt = dt
        self.exp_name = exp_name
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.physics = KuramotoSivashinsky()
        self.final_reward_factor = final_reward_factor
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()
        self.trajectory = []

    def reset(self):
        self.step_idx = 0
        self.trajectory=[]
        self.physics = KuramotoSivashinsky()
        self.gt_forces = FieldEffect(CenteredGrid(0, **self.domain_dict), ['velocity'])
        self.init_state = CenteredGrid(ks_initial, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)
        # prepare goal state
        state = copy.deepcopy(self.init_state)
        a = [state]
        print("\ncalculating goal state\n")
        for _ in tqdm(range(self.step_count)):
            state = self._step_sim(state.vecotr['x'], (self.gt_forces,))
            a.append(state)
        # trajectory = field.stack(a, spatial('time'), Box(time=self.step_count * self.dt))
        # vis.show(trajectory.vector[0], aspect='auto', size=(8, 6))
        # quit()
        self.goal_state = state
        # init reference states
        if self.test_mode:
            self.gt_state = copy.deepcopy(self.init_state)
        return self._build_obs()

    def step(self, actions: np.ndarray):
        self.step_idx += 1
        self.actions = actions.reshape(self.cont_state.data._native.shape)
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        forces = self.actions
        forces_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict, extrapolation=extrapolation.BOUNDARY), ['velocity_effect'])
        self.cont_state = self._step_sim(self.cont_state.vector['x'], (forces_effect,))

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

        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        if not self.test_mode:
            self.test_mode = True
            self.gt_state = copy.deepcopy(self.init_state)
        # super(KS3EnvGym, self).render()
        # we use last_render in the end of control program in experiment

    def last_render(self):
        temp_t = field.stack(self.physics.trajectory, spatial('time'),
                             Box(time=self.step_count * self.dt))  # time=len(trajectory)* dt
        vis.show(temp_t.vector[0], aspect='auto', size=(8, 6))

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
