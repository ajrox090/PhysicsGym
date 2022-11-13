import numpy as np
from typing import Optional, Tuple
from stable_baselines3.common.policies import BasePolicy

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv

from src.env.PhysicsGym import PhysicsGym


class PhysicsAgent(BaseAlgorithm):

    def __init__(self, env: PhysicsGym, policy: BasePolicy = MlpPolicy, lr: float = 0.0001):
        super().__init__(policy=policy, env=env, learning_rate=lr)
        self.env = env

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

        raise NotImplementedError

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
              n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        """ MPC is an optimization algorithm, it doesn't learn anything """
        return self
