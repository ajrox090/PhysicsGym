import copy
import phi.math
import numpy as np
from matplotlib import pyplot as plt

from phi.field import CenteredGrid
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.PhysicsGym import PhysicsGym
from src.env.physics.heat import Heat


class HeatPhysicsGym(PhysicsGym):
    def __init__(self,
                 domain: int = 5,
                 dx: float = 0.25,
                 step_count: int = 1000,
                 domain_dict=None,
                 dt: float = 0.01,
                 diffusivity: int = 0.3):
        super(HeatPhysicsGym, self).__init__(domain, dx, dt, step_count,
                                             domain_dict, reward_rms=RunningMeanStd())

        self.final_state = None
        self.actions_grid_trans = None
        self.diffusivity = diffusivity
        self.physics = Heat(diffusivity=diffusivity)
        self.reward = []
        self.previous_rew = []

    def reset(self):
        self.step_idx = 0
        if self.reward is not None:
            self.previous_rew = copy.deepcopy(self.reward)
            self.reward = []
        if self.init_state is None:
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
        # if self.step_idx % (self.step_count - 1) == 0:
        #     self.render()

        # post-processing
        self.step_idx += 1
        obs = self._build_obs()
        rew = self._build_reward(obs)
        self.reward_rms.update(rew)
        rew = (rew - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)
        done = np.full((1,), self.step_idx == self.step_count)
        if self.step_idx == self.step_count:
            self.final_state = copy.deepcopy(self.cont_state)
        info = {'rew_normalized': rew}
        rew = np.sum(rew, axis=0)
        self.reward.append(rew)
        return obs, rew, done, info

    def render(self, mode: str = 'final', title: str = 'HeatPhysicsGym') -> None:
        x = np.arange(0, self.domain, self.dx)
        plt.tick_params(axis='x', which='minor', length=10)
        plt.grid(True, linestyle='--', which='both')
        plt.plot(x, self.init_state.data.native("vector,x")[0], label='init state')
        plt.plot(x, self.actions_grid_trans.data.native("vector,x")[0], label='action')
        if mode == 'final':
            plt.plot(x, self.final_state.data.native("vector,x")[0], label='final state')
        elif mode == 'cont':
            plt.plot(x, self.cont_state.data.native("vector,x")[0], label='cont state')
        plt.plot(x, self.reference_state_np, label='target state')
        plt.xlim(0, self.domain)
        plt.ylim(-3, 3)
        plt.legend()
        plt.title(title)
        plt.show()

    def _build_obs(self) -> np.ndarray:
        return self.cont_state.data.native("vector,x")[0]

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        return -np.sum((obs - self.reference_state_np) ** 2 / self.N, axis=-1)
