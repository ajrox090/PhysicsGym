import numpy as np
from typing import Optional, Tuple

from src.agent.PhysicsAgent import PhysicsAgent
from src.env.PhysicsGym import PhysicsGym


class RandomAgent(PhysicsAgent):

    def __init__(self, env: PhysicsGym):
        super().__init__(env=env)

    def predict(self, observation: np.ndarray, state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False) \
            -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return [self.env.action_space.sample()]
