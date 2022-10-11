import copy
from typing import Optional, Union, Tuple

import numpy as np
from phi import math
from phi.field import CenteredGrid
from phi.physics._effect import FieldEffect
from scipy.optimize import Bounds, minimize
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule
from stable_baselines3.ppo import MlpPolicy

from src.env.PhysicsGym import PhysicsGym


class MPCAgent(BaseAlgorithm):

    def __init__(self, env: PhysicsGym, learning_rate: Union[float, Schedule],
                 tensorboard_log: Optional[str] = None,
                 ph: int = 10,  # prediction horizon
                 u_min: float = -1,  # lower bound for control
                 u_max: float = 1,  # upper bound for control
                 N: int = 8,  # domain / resolution
                 ):
        super().__init__(policy=MlpPolicy, env=env, learning_rate=learning_rate, tensorboard_log=tensorboard_log)

        self.ph = ph
        self.env = env
        self.nt2 = env.step_count + 1
        self.u_min = u_min
        self.u_max = u_max

        # ground truth trajectories
        self.gt_trajectory = np.array([np.zeros(self.env.N).reshape(self.env.N, 1) for x in range(self.env.step_count)])

        # box constraints u_min <= u <= u_max
        self.bounds = Bounds(self.u_min * np.ones(self.p * self.env.N, dtype=float),
                             self.u_max * np.ones(self.p * self.env.N, dtype=float))

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
              n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        """ MPC is an optimization algorithm, it doesn't learn anything """
        return self

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # initial guess for first optimization problem which is 0
        u0 = [np.zeros(self.env.N) for _ in range(self.ph)]

        y_ = copy.deepcopy(self.env.init_state.data.native("vector,x")[0])
        ypf_ = copy.deepcopy(self.env.init_state)
        u_ = np.zeros((self.nt2, self.env.N))
        y_traj = [ypf_]

        for i in range(self.nt2):

            # determine maximum entry of reference trajectory
            # if ie - i < p, then the remaining entries are
            # constant and identical to the last given one
            ie = np.min([self.nt2, i + self.ph + 1])

            # call optimizer
            res = minimize(lambda utmp: self.cost_function(utmp, y_, self.gt_trajectory[i:ie]),
                           u0, method='SLSQP', bounds=self.bounds)

            # retrieve first entry of u and apply it to the plant
            u_[i] = res.x[0]
            # apply above control on environment
            if i < self.nt2 - 1:
                u0_ = math.tensor(u_[i].reshape(y_.shape), ypf_.shape)
                f_ = FieldEffect(CenteredGrid(u0_, **self.env.domain_dict), ['velocity'])
                ypf_ = self.env.physics.step(ypf_, dt=self.env.dt, effects=(f_,))
                y_traj.append(copy.deepcopy(ypf_))
                y_ = ypf_.data.native("vector,x")[0]

            # update initial guess
            u0[:-1] = res.x[1:]
            u0[-1] = res.x[-1]

        return y_traj

    def cost_function(self, u_: np.ndarray, y_0: np.array, y_gt: np.ndarray):
        """
            step1 -> update the environment for finite number of prediction horizon in the presence of control element u_
            step2 -> extract target trajectories from y_gt_native of size = prediction horizon
            step3 -> calculate weighted difference
        """

        # step1: update the environment for the prediction horizon p using given control u
        u_ = u_.reshape(self.ph, self.env.N)
        y_ = [y_0]
        for i in range(self.ph):
            y = CenteredGrid(math.tensor(y_0, self.env.init_state.shape), **self.env.domain_dict)

            u0_ = math.tensor(u_[i].reshape(y_0.shape), self.env.init_state.shape)
            f_ = FieldEffect(CenteredGrid(u0_, **self.env.domain_dict), ['velocity'])

            y = self.env.physics.step(y, dt=self.env.dt, effects=(f_,))
            y_0 = y.data.native("vector,x")[0]
            y_.append(y_0)
        y_ = np.array(y_)

        # step2: select the reference trajectories
        # fill up reference trajectory if necessary
        y_ref = np.zeros(y_.shape)
        y_ref[:y_gt.shape[0]] = y_gt

        # step3: calculate weighted difference between trajectories
        dy = y_ - y_ref
        dyQ = 0
        # for ii in range(dy.shape[1]):
        for ii in range(dy.shape[1]):
            dyQ += np.power(dy[:, ii], 2)

        return self.env.dt * np.sum(dyQ)
