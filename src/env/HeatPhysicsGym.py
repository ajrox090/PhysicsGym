import copy
import numpy as np

from phi.field import CenteredGrid

from src.env.physics.heat import Heat
from src.env.PhysicsGym import PhysicsGym


class HeatPhysicsGym(PhysicsGym):
    def __init__(self,
                 domain: int = 5,
                 dx: float = 0.25,
                 step_count: int = 1000,
                 domain_dict=None,
                 dt: float = 0.01,
                 diffusivity: int = 0.3,
                 dxdt: int = 100,
                 saveFig: bool = False,
                 title: str = "experiment1",
                 plotFolder: str = "plots"):
        super(HeatPhysicsGym, self).__init__(domain, dx, dt, step_count, domain_dict,
                                             dxdt=dxdt, saveFig=saveFig, title=title,
                                             plotFolder=plotFolder)

        self.forces = None
        self.physics = Heat(diffusivity=diffusivity)
        self.reset()

    def reset(self):
        self.step_idx = 0

        self.init_state = CenteredGrid(self.simpleUniformRandom, **self.domain_dict)
        # self.init_state = CenteredGrid(self.justOnes, **self.domain_dict)

        self.cont_state = copy.deepcopy(self.init_state)
        self.reference_state_np = np.zeros(self.N).reshape(self.N, 1)

        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data.native("vector,x")[0]

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        rew = -np.sum((obs - self.reference_state_np) ** 2 / self.N, axis=-1)
        rew = np.sum(rew, axis=0)
        return rew
