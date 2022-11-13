import copy
import numpy as np

from phi.field import CenteredGrid
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.PhysicsGym import PhysicsGym
from src.env.physics.burgers import Burgers


class BurgersPhysicsGym(PhysicsGym):
    def __init__(self, domain: int = 5, dx: float = 0.25, step_count: int = 1000, domain_dict=None, dt: float = 0.01,
                 viscosity: int = 0.3, diffusion_substeps: int = 1,
                 dxdt: int = 100, saveFig: bool = False, title: str = "experiment1", plotFolder: str = "plots",
                 effects_label: str = "effect", xlim: int = 0, ylim: int = 3):
        super(BurgersPhysicsGym, self).__init__(domain, dx, dt, step_count,
                                                domain_dict, dxdt=dxdt, saveFig=saveFig,
                                                title=title, plotFolder=plotFolder,
                                                effects_label=effects_label,
                                                xlim=xlim, ylim=ylim)

        self.physics = Burgers(default_viscosity=viscosity, diffusion_substeps=diffusion_substeps)
        self.reset()
        self.reward_rms = RunningMeanStd()

    def reset(self):
        self.step_idx = 0
        self.init_state = CenteredGrid(self.simpleNormalDistribution, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)
        self.reference_state_np = np.zeros(self.N).reshape(self.N, 1)
        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data.native("vector,x")[0]

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        rew = -np.sum((obs - self.reference_state_np) ** 2 / self.N, axis=-1)
        rew = np.sum(rew, axis=0)
        return rew
