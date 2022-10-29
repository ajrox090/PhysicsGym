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
                                                shape=self._get_obs_shape())
        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32,
                                           shape=self._get_act_shape())

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
        obs = self._build_obs()
        rew = self._build_reward(obs)
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
    def _get_obs_shape(self):
        return self.N,

    def _get_act_shape(self):
        return 1,

    def _build_obs(self):
        raise NotImplementedError

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # conversion methods for actions
    def _action_transform(self, alpha):
        # initialize a normal distribution with frozen in mean=-1, std. dev.= 1
        rv = norm(loc=0.5, scale=0.2)
        x = np.arange(0, self.domain, self.dx)
        return alpha * rv.pdf(x) / 2

    def scalar_action_to_forces(self, actions: np.ndarray, label: str = "effect"):
        actions_transformed = self._action_transform(actions[0]).reshape(
            self.cont_state.data.native("x,vector").shape[0])
        return FieldEffect(CenteredGrid(math.tensor(actions_transformed, self.cont_state.shape), **self.domain_dict),
                           [label])

    def b_scalar_action_to_forces(self, actions: np.ndarray, label: str = "effect"):
        return FieldEffect(CenteredGrid(math.tensor(self._action_transform(actions[0]).reshape(
            self.cont_state.data.native("x,vector").shape), self.cont_state.shape), **self.domain_dict),
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
        return tensor(np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 0.5, self.N), x.shape[0])

    @staticmethod
    def simpleNormalDistribution(x):
        result = tensor(norm.pdf(x.native("vector,x")[0], np.random.uniform(0, 1.0), 0.2), x.shape[0]) / 1.2
        return result

    def justOnes(self, x):
        return tensor([1.0 for _ in range(self.N)], x.shape[0])

    def justRandom(self, x):
        return tensor([1.0 if i < self.N / 2 else 0.0 for i in range(self.N)], x.shape[0])

    def simpleGauss(self, x):
        N = x.native("vector,x")[0].shape
        xshape = x.shape
        # n = np.random.random(12)
        # leftloc = math.random_uniform(xshape, low=n[0], high=n[1])
        # leftamp = math.random_uniform(xshape, low=n[2], high=n[3])
        # leftsig = math.random_uniform(xshape, low=n[4], high=n[5])
        # rightloc = math.random_uniform(xshape, low=n[6], high=n[7])
        # rightamp = math.random_uniform(xshape, low=-n[8], high=n[9])
        # rightsig = math.random_uniform(xshape, low=n[10], high=n[11])

        leftloc = math.random_uniform(xshape, low=0.2, high=0.4)
        leftamp = math.random_uniform(xshape, low=0, high=3)
        leftsig = math.random_uniform(xshape, low=0.05, high=0.15)
        rightloc = math.random_uniform(xshape, low=0.6, high=0.8)
        rightamp = math.random_uniform(xshape, low=-3, high=0)
        rightsig = math.random_uniform(xshape, low=0.05, high=0.15)

        left = leftamp * math.exp(-0.5 * (x - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * math.exp(-0.5 * (x - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        return result
