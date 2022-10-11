import copy

import gym
import numpy as np
from phi import math
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from typing import Optional, Tuple

from phi.field import Field, CenteredGrid
from phi.math import tensor, inf
from phi.physics._effect import FieldEffect
from stable_baselines3.common.running_mean_std import RunningMeanStd
from tqdm import tqdm


class PhysicsGym(gym.Env):
    def __init__(self,
                 domain, dx,
                 dt, step_count, domain_dict,
                 reward_rms: [Optional] = RunningMeanStd()):
        super(PhysicsGym, self).__init__()

        # initialization and step variables
        self.dx = dx
        self.domain = domain
        self.N = int(domain / dx)
        self.dt = dt
        self.step_count = step_count
        self.domain_dict = domain_dict

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
        self.reference_state_phi = None

        self.ep_idx = 0
        self.step_idx = 0
        self.actions = None
        self.reward_rms = reward_rms
        self.reward_range = (-float('inf'), float('inf'))

    def reset(self):
        """ set: - initial state, cont state and reference state
            returns: observation """
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        """ transform(x) := domain(d=1) -> domain(x)
            # 1) pre-processing: 1-d actions + transform(actions)
            # 2) update environment: step env
            # 3) post-processing:
            #       - transform(observations)
            #       - compute(rewards)"""
        raise NotImplementedError

    def render(self, mode: str = 'live'):
        raise NotImplementedError

    def _step_sim(self, in_state: Field,
                  effects: Tuple[FieldEffect, ...]) -> Field:
        return self.physics.step(in_state, dt=self.dt, effects=effects)

    def _get_obs_shape(self):
        return self.N,

    def _get_act_shape(self):
        return 1,

    def _build_obs(self):
        raise NotImplementedError

    def _build_reward(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def action_transform(self, alpha):
        # initialize a normal distribution with frozen in mean=-1, std. dev.= 1
        rv = norm(loc=self.domain/2, scale=0.2)
        x = np.arange(0, self.domain, self.dx)
        return alpha * rv.pdf(x) / 2

    def simpleGauss(self, x):
        return tensor(np.random.uniform(0, 0.5, self.N), x.shape[0])

    def mpc_cost(self, u_: np.ndarray, y_0_native: np.array, y_gt_native: np.ndarray):
        """
            step1 -> update the environment for finite number of prediction horizon in the presence of control element u_
            step2 -> extract target trajectories from y_gt_native of size = prediction horizon
            step3 -> calculate weighted difference
        """

        # step1: update the environment for the prediction horizon p using given control u
        u_ = u_.reshape(self.p, self.N)
        y0_ = copy.deepcopy(y_0_native)
        y_ = [y0_]
        for i in range(self.p):
            y00_ = math.tensor(y0_, self.init_state.shape)
            y0_ = CenteredGrid(y00_, **self.domain_dict)
            u0_ = math.tensor(u_[i].reshape(y_0_native.shape), self.init_state.shape)
            f_ = FieldEffect(CenteredGrid(u0_, **self.domain_dict), ['velocity'])
            y0_ = self.physics.step(y0_, dt=self.dt, effects=(f_,))
            y0_ = y0_.data.native("vector,x")[0]
            y_.append(y0_)
        y_ = np.array(y_)

        # step2: select the reference trajectories
        # fill up reference trajectory if necessary
        y_ref = np.zeros(y_.shape)
        y_ref[:y_gt_native.shape[0]] = y_gt_native

        # step3: calculate weighted difference between trajectories
        dy = y_ - y_ref
        dyQ = 0
        # for ii in range(dy.shape[1]):
        for ii in range(dy.shape[1]):
            dyQ += np.power(dy[:, ii], 2)

        return self.dt * np.sum(dyQ)

    def mpc(self):
        self.p = 10  # Prediction horizon on coarse grid
        T = 1.0  # New final time
        nt2 = round(T / self.dt) + 1  # number of time steps on coarser grid for SUR
        u_min = -10.0  # lower bound for control
        u_max = 10.0  # upper bound for control

        # ground truth trajectories
        gt_trajectory = np.array([np.zeros(self.N).reshape(self.N, 1) for x in range(self.step_count)])

        # intial guess for first optimization problem
        u0 = [np.zeros(self.N).reshape(self.N, 1) for _ in range(self.p)]
        # box constraints u_min <= u <= u_max
        bounds = Bounds(u_min * np.ones(self.p * self.N, dtype=float), u_max * np.ones(self.p * self.N, dtype=float))

        y_ = copy.deepcopy(self.init_state.data.native("vector,x")[0])
        ypf_ = copy.deepcopy(self.init_state)
        u_ = np.zeros((nt2, self.N))
        y_traj = [ypf_]
        for i in tqdm(range(nt2)):

            # determine maximum entry of reference trajectory
            # if ie - i < p, then the remaining entries are
            # constant and identical to the last given one
            ie = np.min([nt2, i + self.p + 1])

            # call optimizer
            res = minimize(lambda utmp: self.mpc_cost(utmp, y_, gt_trajectory[i:ie]), u0, method='SLSQP', bounds=bounds)

            # retrieve first entry of u and apply it to the plant
            u_[i] = res.x[0]
            # apply above control on environment
            if i < nt2 - 1:
                u0_ = math.tensor(u_[i].reshape(ypf_.data.native("vector,x").shape), ypf_.shape)
                f_ = FieldEffect(CenteredGrid(u0_, **self.domain_dict), ['velocity'])
                ypf_ = self.physics.step(ypf_, dt=self.dt, effects=(f_,))
                y_traj.append(copy.deepcopy(ypf_))
                y_ = ypf_.data.native("vector,x")[0]

            # update initial guess
            u0[:-1] = res.x[1:]
            u0[-1] = res.x[-1]

        return y_traj













