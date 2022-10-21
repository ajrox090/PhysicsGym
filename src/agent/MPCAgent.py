import numpy as np
from typing import Optional, Tuple, Type, Union
from scipy.optimize import Bounds, minimize

import phi
from phi import math
from phi.field import CenteredGrid, Grid, Field
from phi.physics._effect import FieldEffect
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule
from tqdm import tqdm

from src.env.PhysicsGym import PhysicsGym


class MPCAgent(BaseAlgorithm):

    def __init__(self, env: PhysicsGym,
                 ph: int = 10,  # prediction horizon,
                 u_max: float = 1.0,  # max control input
                 u_min: float = -1.0,  # min control input
                 u0: np.ndarray = None,  # initial control
                 ):
        super().__init__(policy=MlpPolicy, env=env, learning_rate=1.0)
        self.env = env
        self.ph = ph
        # self.Q = np.array([0.7 if i < 4 else 1.0 for i in range(env.N)])

        # self.u0 = np.random.uniform(u_min, u_max, self.ph)
        self.u0 = np.zeros(self.ph)

        self.bounds = tuple(zip([u_min for _ in range(self.ph)], [u_max for _ in range(self.ph)]))

        self.ref_state = env.reference_state_np.flatten()
        self.shape_nt_state = env.init_state.data.native("x,vector").shape[0]
        self.shape_phi_state = env.init_state.shape

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        res = minimize(lambda u: self.cost_function(u, observation),
                       self.u0, method='SLSQP', bounds=self.bounds,
                       options={'eps': 1e-1})

        self.u0[:-1] = res.x[1:]
        self.u0[-1] = res.x[-1]

        if not res.success:
            raise Warning("Failed to find minimum")

        print(f'{res.x} : {res.fun}')
        return [res.x[0]]

    def cost_function(self, u_, y0_):

        y0_ = CenteredGrid(phi.math.tensor(y0_, self.shape_phi_state), **self.env.domain_dict)

        for ii in range(self.ph):
            y0_ = self.env.step_physics(in_state=y0_,
                                        effects=(self.env.scalar_action_to_forces([u_[ii]]),))

        loss = np.sum(((y0_.data.native("vector,x")[0] - self.env.reference_state_np) ** 2) / self.env.N, axis=-1)
        loss = np.sum(loss, axis=0)
        # loss = np.sum((y0_.data.native("vector,x")[0]) ** 2 / self.env.N)
        # loss = np.sum((y0_ - self.env.reference_state_np) ** 2) / self.env.N
        # print(f"{u_},{loss}")
        return loss

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
              n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        """ MPC is an optimization algorithm, it doesn't learn anything """
        return self
