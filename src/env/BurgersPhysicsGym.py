import copy
import numpy as np
from matplotlib import pyplot as plt

import phi.math
from phi.field import CenteredGrid
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.PhysicsGym import PhysicsGym
from src.env.physics.burgers import Burgers


class BurgersPhysicsGym(PhysicsGym):
    def __init__(self,
                 domain: int = 5,
                 dx: float = 0.25,
                 step_count: int = 1000,
                 domain_dict=None,
                 dt: float = 0.01,
                 viscosity: int = 0.3,
                 final_reward_factor: float = 32):
        super(BurgersPhysicsGym, self).__init__(domain, dx, dt, step_count, domain_dict,
                                                final_reward_factor, reward_rms=RunningMeanStd())

        self.initial_state = None
        self.final_state = None
        self.actions_grid_trans = None
        self.viscosity = viscosity
        self.physics = Burgers(default_viscosity=viscosity, diffusion_substeps=1)

    def reset(self):
        self.step_idx = 0

        self.init_state = CenteredGrid(self.simpleGauss, **self.domain_dict)
        self.cont_state = copy.deepcopy(self.init_state)

        self.reference_state_np = np.zeros(self.N).reshape(self.N, 1)
        self.reference_state_phi = CenteredGrid(self.reference_state_np.reshape(-1)[0], **self.domain_dict)
        return self._build_obs()

    def step(self, actions: np.ndarray):

        # prepare actions
        actions = self.action_transform(actions[0])
        self.actions = actions.reshape(self.cont_state.data.native("x,vector").shape[0])
        actions_tensor = phi.math.tensor(self.actions, self.cont_state.shape)
        self.actions_grid_trans = CenteredGrid(actions_tensor, **self.domain_dict)
        forces_effect = FieldEffect(self.actions_grid_trans, ['temperature_effect'])

        # step environment
        self.cont_state = self._step_sim(self.cont_state, (forces_effect,))

        # visualize
        if self.step_idx % int((self.step_count - 1) / 10) == 0:
            self.render()

        # post-processing
        self.step_idx += 1
        obs = self._build_obs()
        rew = self._build_reward(obs)
        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        done = np.full((1,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.final_state = copy.deepcopy(self.cont_state)
        info = {'rew_unnormalized': rew}

        return obs, np.sum(rew, axis=0), done, info

    def render(self, mode: str = 'live') -> None:
        x = np.arange(0, self.domain, self.dx)
        plt.tick_params(axis='x', which='minor', length=10)
        plt.grid(True, linestyle='--', which='both')
        plt.plot(x, self.init_state.data.native("vector,x")[0], label='init state')
        plt.plot(x, self.actions_grid_trans.data.native("vector,x")[0], label='action')
        plt.plot(x, self.cont_state.data.native("vector,x")[0], label='cont state')
        plt.plot(x, self.reference_state_np, label='final state')
        plt.xlim(0, self.domain)
        plt.ylim(-3, 3)
        plt.legend()
        plt.show()

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data.native("vector,x")[0]

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        return -np.sum((obs - self.reference_state_np) ** 2 / self.N, axis=-1)
