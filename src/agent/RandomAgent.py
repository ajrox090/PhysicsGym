import numpy as np
from typing import Optional, Union, Tuple

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule


class RandomAgent(BaseAlgorithm):

    def __init__(self, env: Union[GymEnv, str, None]):
        super().__init__(policy=MlpPolicy, env=env, learning_rate=1.0)

    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
              n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        return self

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return [self.env.action_space.sample()]
