import numpy as np
from typing import Optional, Tuple
from scipy.optimize import Bounds, minimize

import phi
from phi import math
from phi.field import CenteredGrid, Grid, Field
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule
from tqdm import tqdm

from src.env.PhysicsGym import PhysicsGym


class MPCAgent(BaseAlgorithm):

    def __init__(self, env: PhysicsGym,
                 ph: int = 10,  # prediction horizon,
                 init_control: np.ndarray = None,  # initial control
                 u_max: float = 1.0,  # max control input
                 u_min: float = -1.0,  # min control input
                 ):
        super().__init__(policy=MlpPolicy, env=env, learning_rate=1.0)
        self.env = env
        self.ph = ph
        # self.init_control = np.random.uniform(u_min, u_max, self.ph)
        self.init_control = np.zeros(self.ph)
        # self.init_control = np.ones(self.ph)
        # self.Q = [(1.0 if i < self.env.N // 2 else 0.1) for i in range(self.env.N)]

        self.bounds = tuple(zip([u_min for _ in range(self.ph)], [u_max for _ in range(self.ph)]))
        print(self.bounds)
        # self.bounds = Bounds(u_min, u_max)

        # self.constraints = ({'type': 'ineq', 'fun': lambda x: np.sum(x) - 9.0})

        self.ref_states_np = np.array([env.reference_state_np.flatten() for _ in range(self.ph)])
        self.shape_nt_state = env.init_state.data.native("x,vector").shape[0]
        self.shape_phi_state = env.init_state.shape

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        res = minimize(lambda cont: self.cost_function(cont, observation, self.ref_states_np),
                       self.init_control, method='SLSQP', bounds=self.bounds)  # Nelder-Mead
        # res.x = self.env.dt * res.x

        self.init_control[:-1] = res.x[1:]
        self.init_control[-1] = res.x[-1]
        if not res.success:
            raise Warning("Failed to find minimum")
        # else:
        #     print("success")
        return [res.x[0]]

    def cost_function(self, actions: np.ndarray, curr_state: np.ndarray, ref_states: np.ndarray):
        """
            step1 -> update the environment for finite number of prediction horizon in the presence of control element u_
            step2 -> extract target trajectories from y_gt_native of size = prediction horizon
            step3 -> calculate weighted difference
        """
        state = CenteredGrid(phi.math.tensor(curr_state, self.shape_phi_state), **self.env.domain_dict)
        states = []

        for i in tqdm(range(self.ph)):
            action = FieldEffect(CenteredGrid(
                phi.math.tensor(self.env.action_transform(actions[i]).reshape(self.shape_nt_state),
                                self.shape_phi_state),
                **self.env.domain_dict), ['effect'])
            state = self.env.step_physics(in_state=state, effects=(action,))
            states.append(state.data.native("vector,x")[0])

        dy = np.array(states) - ref_states
        dyQ = np.zeros(dy.shape[0], dtype=float)

        for ii in range(dy.shape[1]):
            dyQ += np.power(dy[:, ii], 2)

        # return self.env.dt * np.sum(dyQ)
        return np.sum(dyQ) / self.env.N

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
              n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        """ MPC is an optimization algorithm, it doesn't learn anything """
        return self
