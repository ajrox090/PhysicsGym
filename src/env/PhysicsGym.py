import gym
import numpy as np
from typing import Tuple
from scipy.stats import norm
from matplotlib import pyplot as plt

from phi import math
from phi.math import tensor, inf
from phi.flow_1 import FieldEffect
from phi.field import Field, CenteredGrid


class PhysicsGym(gym.Env):
    def __init__(self,
                 domain, dx,
                 dt, step_count, domain_dict,
                 dxdt: int = 100,
                 saveFig: bool = False,
                 title: str = "experiment1",
                 plotFolder: str = "plots",
                 effects_label: str = "effect",
                 xlim: int = 0,
                 ylim: int = 3
                 ):
        super(PhysicsGym, self).__init__()

        # initialization and step variables
        self.dx = dx
        self.dt = dt
        self.dxdt = dxdt  # number of time steps to simulate the environment
        self.title = title
        self.domain = domain
        self.saveFig = saveFig
        self.plotFolder = plotFolder
        self.N = int(domain / dx)
        self.step_count = step_count
        self.domain_dict = domain_dict
        self.effects_label = effects_label

        self.observation_space = gym.spaces.Box(low=-inf, high=inf, dtype=np.float32,
                                                shape=self.obs_shape())
        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32,
                                           shape=self.action_shape())

        # variables specific to problem
        self.physics = None

        # states and forces
        self.init_state = None
        self.cont_state = None
        self.reference_state_np = None

        self.ep_idx = 0
        self.step_idx = 0
        self.forces = None
        self._render = False

        # plot params
        self.xlim = xlim
        self.ylim = ylim

    def reset(self):
        """ set: - initial state, cont state and reference state
            returns: observation """
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        """ 1. transform actions to apply to the physics environment
            2. update the ennvironment
            3. visualize new state of environment
            4. calculate new observations
            5. prepare rewards
            """

        # prepare actions
        self.forces = self.scalar_action_to_forces(actions, label=self.effects_label)
        # self.forces = self.b_scalar_action_to_forces(actions, label=self.effects_label)

        # step environment
        self.cont_state = self.step_physics(self.cont_state, (self.forces,))

        # visualize
        if self._render:
            self.render(xlim=self.xlim, ylim=self.ylim,
                        title=f'Burgers simulation with {[actions[0] if actions[0] > 0 else "uncontrolled"][0]}'
                              f' action at step {self.step_idx}')

        # post-processing
        self.step_idx += 1
        obs = self.build_obs()
        rew = self.build_reward(obs)
        done = np.full((1,), self.step_idx == self.step_count + 1)
        info = {'reward': rew}
        return obs, rew, done, info

    def render(self, mode: str = 'live', title: str = 'PhysicsGym',
               xlim: int = 0, ylim: int = 3) -> None:
        """ initial state and cont state should be initialized using reset at least once,
        before this method is called."""

        x = np.arange(0, self.domain, self.dx)
        plt.tick_params(axis='x', which='minor', length=10)
        plt.grid(True, linestyle='--', which='both')
        # plt.plot(x, self.init_state.data.native("vector,x")[0], label='init state')
        if self.forces is not None:
            plt.plot(x, self.forces_to_numpy(self.forces), label='action')
        plt.plot(x, self.cont_state.data.native("vector,x")[0], label='cont state')
        # plt.plot(x, self.reference_state_np, label='target state')
        plt.xlim(xlim, self.domain)
        plt.ylim(-ylim, ylim)
        plt.legend()
        plt.title(title)
        if self.saveFig:
            plt.savefig(f'{self.plotFolder}/{self.title}/{self.step_idx}.pdf', bbox_inches='tight')
        plt.show()

    def step_physics(self, in_state: Field,
                     effects: Tuple[FieldEffect, ...]) -> Field:
        for i in range(self.dxdt):
            in_state = self.physics.step(in_state, dt=self.dt, effects=effects)
        return in_state

    # helper methods
    def obs_shape(self):
        """ This method defines the shape of the observation space."""
        raise NotImplementedError

    def action_shape(self):
        """ This method defines the shape of the actions space"""
        raise NotImplementedError

    def build_obs(self):
        """ This method defines the observation vector """
        raise NotImplementedError

    def build_reward(self, obs: np.ndarray) -> np.ndarray:
        """ This method defines the calculation of reward for each time step.
        The normalization of rewards should be handled here. """
        raise NotImplementedError

    def action_transform(self, alpha):
        """ This method defines the transformation of actions before applying to Phiflow's field object."""
        raise NotImplementedError

    def scalar_action_to_forces(self, actions: np.ndarray, label: str = "effect"):
        actions_transformed = self.action_transform(actions[0]).reshape(
            self.cont_state.data.native("x,vector").shape[0])
        return FieldEffect(CenteredGrid(math.tensor(actions_transformed, self.cont_state.shape), **self.domain_dict),
                           [label])

    @staticmethod
    def forces_to_numpy(forces: FieldEffect):
        return forces.field.data.native("vector,x")[0]

    # visualization methods
    def enable_rendering(self):
        self._render = True

    def disable_rendering(self):
        self._render = False

    # initial states
    def simpleUniformRandom(self, x):
        return tensor(np.random.uniform(0, 0.5, self.N), x.shape[0])

    @staticmethod
    def simpleNormalDistribution(x):
        result = tensor(norm.pdf(x.native("vector,x")[0], np.random.uniform(0, 1.0), 0.2), x.shape[0]) / 1.2
        return result
