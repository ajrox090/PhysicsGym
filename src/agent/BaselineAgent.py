import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize

import phi
from phi import math
from phi.field import CenteredGrid

from src.agent.PhysicsAgent import PhysicsAgent
from src.env.PhysicsGym import PhysicsGym


class BaselineAgent(PhysicsAgent):

    def __init__(self, env: PhysicsGym, u_max: float = 1.0, u_min: float = -1.0):
        super().__init__(env=env)
        self.u0 = np.array([0, 0])
        self.bounds = tuple(zip([u_min for _ in range(2)], [u_max for _ in range(2)]))

    def predict(self, observation: np.ndarray, state: Optional[np.ndarray] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False) \
            -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        res = minimize(lambda u: self.cost_function(u, observation), self.u0, method='SLSQP', bounds=self.bounds,
                       options={'eps': 1e-1})
        if not res.success:
            raise Warning("Failed to find minimum")
            print(f'{res.x} : {res.fun}')

        return [res.x[0]]

    def cost_function(self, u_, y0_):
        y0_ = self.env.step_physics(in_state=CenteredGrid(phi.math.tensor(y0_, self.shape_phi_state),
                                                          **self.env.domain_dict),
                                    effects=(self.env.scalar_action_to_forces([u_[0]]),))

        loss = np.sum(((y0_.data.native("vector,x")[0] - self.env.reference_state_np) ** 2) / self.env.N, axis=-1)
        loss = np.sum(loss, axis=0)

        return loss
