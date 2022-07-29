import gym
from phi import math, geom, vis
import numpy as np
from phi.geom import Box
from typing import Optional, Tuple, Union, Dict
from phi.field import Field, CenteredGrid, StaggeredGrid, Grid
from phi.math import spatial, batch, channel
from phi.physics._boundaries import Obstacle
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.phiflow.navier_stokes import NavierStokes
from src.util.navier_stokes_util import _get_obs_shape, _get_act_shape

GymEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]


class NavierStokesEnvGym(gym.Env):

    def __init__(self, N,
                 step_count: int = 32,
                 domain_dict=None,
                 dt: float = 0.03,
                 final_reward_factor: float = 32,
                 reward_rms: Optional[RunningMeanStd] = None,
                 ):
        super(NavierStokesEnvGym, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=_get_obs_shape(tuple([domain_dict['x'], domain_dict['y']])))
        self.action_space = gym.spaces.Box(low=-100, high=-50, dtype=np.float32,
                                           shape=_get_act_shape(tuple([domain_dict['x'], domain_dict['y']])))  # , domain_dict['y']])))

        self.N = N
        self.domain_dict = domain_dict
        self.step_count = step_count
        self.step_idx = 0
        self.ep_idx = 0
        self.dt = dt
        self.physics = NavierStokes()
        self.final_reward_factor = final_reward_factor
        self.reward_range = (-float('inf'), float('inf'))
        self.reward_rms = reward_rms
        if self.reward_rms is None:
            self.reward_rms = RunningMeanStd()
        self.actions = None

        # Physics: prepare states, boundary mask based on state and obstacles
        self.init_state = StaggeredGrid((2, 0), **self.domain_dict)
        self.cont_state = self.init_state

        self.pressure = None
        self.vorticity = None
        self.boundary_mask = CenteredGrid(
            Box(x=(-math.INF, 0.5), y=None),
            self.init_state.extrapolation, self.init_state.bounds,
            self.init_state.resolution)

        # cylinder1 = Obstacle(geom.infinite_cylinder(x=N/4, y=N/2, radius=N/16, inf_dim=None))
        cylinder1 = Obstacle(geom.infinite_cylinder(x=20, y=14, radius=5, inf_dim=None))
        self.obstacles = [cylinder1]

    def reset(self) -> np.ndarray:
        """reset: returns numpy array because this is directly used in 'step' return"""
        self.step_idx = 0
        self.cont_state = self.init_state

        return self._build_obs()

    def step(self, actions: np.ndarray):

        # 1. prepare actions
        # self.actions = actions.reshape(self.cont_state.data._native.shape)
        # self.actions = actions.reshape(self.cont_state.at_centers().values.native('x,vector').shape)
        if isinstance(self.cont_state, StaggeredGrid):
            self.actions = actions.reshape(self.cont_state.at_centers().values.native('x,y,vector').shape)
            actions_tensor = math.tensor(self.actions, self.cont_state.at_centers().values.shape)
        elif isinstance(self.cont_state, CenteredGrid):
            self.actions = actions.reshape(self.cont_state.values.native('x,y,vector').shape)
            actions_tensor = math.tensor(self.actions, self.cont_state.values.shape)
        else:
            raise BaseException
        self.step_idx += 1

        actuators = self.actions
        actions_effect = FieldEffect(CenteredGrid(actions_tensor, **self.domain_dict), ['velocity'])
        vis.plot(actions_effect.field)
        vis.show()

        # 2. step env
        self.cont_state, self.pressure, self.vorticity = self._step_sim(in_state=self.cont_state,
                                                                        effects=(actions_effect,))
        # vis.plot([self.cont_state, self.pressure, self.vorticity], title="step: {}".format(self.step_idx),
        #          show_color_bar=False)
        # vis.show()

        # 4. build output: obs:np.array, reward: Float32, done: np.array, info: dict
        obs = self._build_obs()
        done = np.full((1,), self.step_idx == self.step_count)
        # rew = self._build_rew(actuators)
        rew = self._build_drag(self.cont_state, self.pressure)
        info = {'rew_unnormalized': rew, 'actions': np.abs(actuators).sum()}
        return obs, np.sum(rew, axis=0), done, info

    def _build_obs(self) -> np.ndarray:
        # return self.cont_state.data._native
        # shape: N x N x 2
        if isinstance(self.cont_state, StaggeredGrid):
            return self.cont_state.at_centers().values.native('y,x,vector')
        elif isinstance(self.cont_state, CenteredGrid):
            return self.cont_state.values.native('y,x,vector')
        else:
            raise BaseException

    def _step_sim(self, in_state: Field, effects: Tuple[FieldEffect, ...]):
        return self.physics.step(in_state, boundary_mask=self.boundary_mask,
                                 pressure=self.pressure,
                                 obstacles=self.obstacles, velocity_effects=effects)

    def _build_rew(self, actions: np.ndarray) -> np.ndarray:
        reshaped_actions = actions.reshape(actions.shape[0], -1)
        reshaped_actions_sum = -np.sum(reshaped_actions ** 2, axis=-1)
        self.reward_rms.update(reshaped_actions_sum)
        return (reshaped_actions_sum - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)  # normalization

    def _build_drag(self, v: Grid, p: Grid):
        cd = 0.82  # drag coefficient
        drag = cd * p * (v ** 2) / 2
        drag_numpy = drag.values.native('x,y,vector')
        drag_reshaped = drag_numpy.reshape(drag_numpy.shape[0], -1)
        drag_reshaped_sum = -np.sum(drag_reshaped ** 2, axis=-1)
        self.reward_rms.update(drag_reshaped_sum)
        return (drag_reshaped_sum - self.reward_rms.mean) / np.sqrt(self.reward_rms.var) # normalization


    def render(self, mode="human"):
        pass
